# -*- coding: utf-8 -*-
"""main.py

Hybrid airfare estimation API.

1. /predict           -> user gives ONLY source, destination and date;
                         returns ML-estimated fares (per airline + summary)
2. /predict-detailed  -> full manual input (airline, times, stops ...)
3. /compare           -> fetches live flight offers from a flight search API,
                estimates the fare of each offer with the ML model,
                and compares the two prices.

Run with:  uvicorn main:app --reload

Environment variables:
    APIFY_TOKEN   token for the live flight search provider (see below)
"""

import os
import re
from datetime import date, datetime
from typing import Literal, Optional

import joblib
import pandas as pd
import requests
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

app = FastAPI(
    title="Hybrid Flight Fare API",
    description="ML fare estimate + live flight offers + price comparison",
    version="2.0"
)

# Load the saved model pipeline
model = joblib.load("flight_fare_model.pkl")

# ---------------------------------------------------------
# Constants
# ---------------------------------------------------------
STOPS_MAP = {
    "non-stop": 0, "1 stop": 1, "2 stops": 2,
    "3 stops": 3, "4 stops": 4
}

# City names used in the training data -> IATA airport codes
CITY_TO_IATA = {
    "Banglore": "BLR", "Bangalore": "BLR",
    "Delhi": "DEL", "New Delhi": "DEL",
    "Kolkata": "CCU",
    "Chennai": "MAA",
    "Mumbai": "BOM",
    "Cochin": "COK",
    "Hyderabad": "HYD",
}

# Airline IATA code -> airline name used in the training data.
# Airlines that are NOT in the training data (e.g. Akasa Air "QP",
# Air India Express "IX") are not listed here, so the model has never
# seen them and the estimate for those offers is less reliable.
CARRIER_TO_AIRLINE = {
    "6E": "IndiGo",
    "AI": "Air India",
    "SG": "SpiceJet",
    "UK": "Vistara",
    "G8": "GoAir",
    "I5": "Air Asia",
    "9W": "Jet Airways",
    "2T": "Trujet",
}

# ---------------------------------------------------------
# Typical flight profiles (so the user only needs source,
# destination and date). Built once at start-up from flight_data.csv.
# ---------------------------------------------------------
def _to_minutes(duration: str) -> int:
    hours, minutes = 0, 0
    for part in duration.split():
        if "h" in part:
            hours = int(part.replace("h", ""))
        else:
            minutes = int(part.replace("m", ""))
    return hours * 60 + minutes


def load_profiles(path="flight_data.csv"):
    df = pd.read_csv(path).dropna().drop_duplicates()
    jd = pd.to_datetime(df["Date_of_Journey"], format="%d/%m/%Y")
    dep = pd.to_datetime(df["Dep_Time"], format="%H:%M")
    arr = pd.to_datetime(
        df["Arrival_Time"].str.split(" ").str[0], format="%H:%M"
    )
    df["Source"] = df["Source"]
    df["Destination"] = df["Destination"].replace("New Delhi", "Delhi")
    df["month"] = jd.dt.month
    df["day"] = jd.dt.day
    df["dep_hour"], df["dep_min"] = dep.dt.hour, dep.dt.minute
    df["arr_hour"], df["arr_min"] = arr.dt.hour, arr.dt.minute
    df["duration_mins"] = df["Duration"].apply(_to_minutes)
    df["stops"] = df["Total_Stops"].map(STOPS_MAP)

    flight_cols = ["dep_hour", "dep_min", "arr_hour", "arr_min",
                   "duration_mins", "stops"]

    # route -> airline -> most common flight on that route
    route_profiles = {}
    for (src, dst, airline), grp in df.groupby(
            ["Source", "Destination", "Airline"]):
        # keep economy-style airlines only, and ignore very rare ones
        if "Business" in airline or "Premium" in airline or len(grp) < 5:
            continue
        top = grp.groupby(flight_cols).size().idxmax()
        route_profiles.setdefault((src, dst), {})[airline] = dict(
            zip(flight_cols, (int(v) for v in top))
        )

    # (month, day) -> festival information seen in the data
    festival_by_day = {}
    for (m, d), grp in df.groupby(["month", "day"]):
        festival_by_day[(int(m), int(d))] = (
            int(grp["Festival_Flag"].mode().iloc[0]),
            int(grp["Days_To_Nearest_Festival"].median()),
        )
    default_days = int(df["Days_To_Nearest_Festival"].median())
    return route_profiles, festival_by_day, default_days


ROUTE_PROFILES, FESTIVAL_BY_DAY, DEFAULT_FESTIVAL_DAYS = load_profiles()


# ---------------------------------------------------------
# Live flight search provider
# (Amadeus Self-Service was shut down on 17 July 2026, so this uses
#  an Amadeus-compatible Google Flights actor on Apify. To use another
#  provider, only change search_live_offers() below.)
# ---------------------------------------------------------
APIFY_ACTOR_URL = (
    "https://api.apify.com/v2/acts/"
    "kuezi~flight-offers-api/run-sync-get-dataset-items"
)


# ---------------------------------------------------------
# Request models
# ---------------------------------------------------------
class FlightInput(BaseModel):
    airline: str = Field(examples=["IndiGo"])
    date_of_journey: date = Field(examples=["2019-03-24"])
    source: str = Field(examples=["Banglore"])
    destination: str = Field(examples=["Delhi"])
    dep_time: str = Field(
        pattern=r"^([01]\d|2[0-3]):[0-5]\d$", examples=["22:20"]
    )
    arrival_time: str = Field(
        pattern=r"^([01]\d|2[0-3]):[0-5]\d$", examples=["01:10"]
    )
    duration_mins: int = Field(gt=0, le=3000, examples=[170])
    total_stops: Literal[
        "non-stop", "1 stop", "2 stops", "3 stops", "4 stops"
    ] = "non-stop"
    festival_flag: int = Field(default=0, ge=0, le=1)
    days_to_nearest_festival: int = Field(default=0, ge=0)


class CompareInput(BaseModel):
    source: str = Field(examples=["Banglore"])
    destination: str = Field(examples=["Delhi"])
    date_of_journey: date = Field(examples=["2026-12-20"])
    non_stop_only: bool = False
    max_offers: int = Field(default=10, ge=1, le=20)
    festival_flag: int = Field(default=0, ge=0, le=1)
    days_to_nearest_festival: int = Field(default=0, ge=0)


class SimpleInput(BaseModel):
    """The only inputs the user has to type."""
    source: str = Field(examples=["Banglore"])
    destination: str = Field(examples=["Delhi"])
    date_of_journey: date = Field(examples=["2026-12-20"])


# ---------------------------------------------------------
# Helper functions
# ---------------------------------------------------------
def normalize_destination(name: str) -> str:
    """'New Delhi' and 'Delhi' are the same city (as in training)."""
    return "Delhi" if name == "New Delhi" else name


def build_features(
    airline, source, destination, journey_date,
    dep_hour, dep_min, arr_hour, arr_min,
    duration_mins, stops, festival_flag, days_to_festival
) -> pd.DataFrame:
    """Build one row with exactly the columns used during training."""
    return pd.DataFrame([{
        "Festival_Flag": festival_flag,
        "Days_To_Nearest_Festival": days_to_festival,
        "day": journey_date.day,
        "month": journey_date.month,
        "weekday": journey_date.weekday(),
        "dep_hour": dep_hour,
        "dep_min": dep_min,
        "arr_hour": arr_hour,
        "arr_min": arr_min,
        "duration_mins": duration_mins,
        "stops": stops,
        "Airline": airline,
        "Source": source,
        "Destination": normalize_destination(destination)
    }])


def iso_duration_to_minutes(duration: str) -> int:
    """Convert an ISO-8601 duration such as 'PT2H50M' into minutes."""
    match = re.fullmatch(
        r"P(?:(\d+)D)?T?(?:(\d+)H)?(?:(\d+)M)?", duration or ""
    )
    if not match:
        raise ValueError(f"Unrecognised duration: {duration}")
    days, hours, minutes = (int(x) if x else 0 for x in match.groups())
    return days * 1440 + hours * 60 + minutes


def search_live_offers(origin_iata, destination_iata, journey_date,
                       non_stop_only, max_offers):
    """Return a list of live flight offers (Amadeus-style JSON)."""
    token = os.getenv("APIFY_TOKEN")
    if not token:
        raise HTTPException(
            status_code=503,
            detail="APIFY_TOKEN environment variable is not set."
        )

    run_input = {
        "endpoint": "flight-offers",
        "originLocationCode": origin_iata,
        "destinationLocationCode": destination_iata,
        "departureDate": journey_date.isoformat(),
        "adults": 1,
        "currencyCode": "INR",
        "max": max_offers,
    }
    if non_stop_only:
        run_input["nonStop"] = True

    try:
        response = requests.post(
            APIFY_ACTOR_URL,
            json=run_input,
            headers={"Authorization": f"Bearer {token}"},
            timeout=120,
        )
        response.raise_for_status()
        items = response.json()
    except requests.RequestException as error:
        raise HTTPException(
            status_code=502,
            detail=f"Flight search API failed: {error}"
        )

    # Items are normally one flight-offer each; also accept a
    # {"data": [...]} wrapper just in case.
    offers = []
    for item in items if isinstance(items, list) else [items]:
        if isinstance(item, dict) and "data" in item:
            offers.extend(item["data"])
        elif isinstance(item, dict) and "itineraries" in item:
            offers.append(item)
    return offers


def offer_to_features(offer, source, destination, festival_flag,
                      days_to_festival):
    """Turn one live offer into model features + readable details."""
    itinerary = offer["itineraries"][0]
    segments = itinerary["segments"]

    first_dep = datetime.fromisoformat(segments[0]["departure"]["at"])
    last_arr = datetime.fromisoformat(segments[-1]["arrival"]["at"])

    carriers = {seg["carrierCode"] for seg in segments}
    if len(carriers) > 1:
        airline = "Multiple carriers"
    else:
        code = next(iter(carriers))
        airline = CARRIER_TO_AIRLINE.get(code)
    known_airline = airline is not None

    duration_mins = iso_duration_to_minutes(itinerary["duration"])
    stops = len(segments) - 1

    features = build_features(
        airline or "Unknown", source, destination, first_dep.date(),
        first_dep.hour, first_dep.minute,
        last_arr.hour, last_arr.minute,
        duration_mins, stops, festival_flag, days_to_festival
    )

    details = {
        "airline_code": "/".join(sorted(carriers)),
        "airline_in_training_data": known_airline,
        "departure": first_dep.isoformat(),
        "arrival": last_arr.isoformat(),
        "duration_mins": duration_mins,
        "stops": stops,
    }
    return features, details


def verdict(live_price: float, ml_price: float) -> str:
    """Compare ML estimate vs live price (10% tolerance)."""
    diff_pct = (ml_price - live_price) / live_price * 100
    if diff_pct < -10:
        return "ML estimate is lower than the live fare"
    if diff_pct > 10:
        return "ML estimate is higher than the live fare"
    return "ML estimate is close to the live fare"


# ---------------------------------------------------------
# Endpoints
# ---------------------------------------------------------
@app.get("/")
def home():
    return {
        "message": "Hybrid Flight Fare API is running",
        "documentation": "/docs"
    }


@app.get("/health")
def health():
    return {"status": "running"}


@app.post("/predict")
def predict_fare(request: SimpleInput):
    """User gives only source, destination and date."""
    source = request.source
    destination = normalize_destination(request.destination)

    if source == destination:
        raise HTTPException(
            status_code=400,
            detail="Source and destination must be different."
        )

    airlines = ROUTE_PROFILES.get((source, destination))
    if not airlines:
        available = sorted(f"{s} -> {d}" for s, d in ROUTE_PROFILES)
        raise HTTPException(
            status_code=400,
            detail={"message": "Route not available in training data.",
                    "available_routes": available}
        )

    journey = request.date_of_journey
    festival_flag, days_to_festival = FESTIVAL_BY_DAY.get(
        (journey.month, journey.day), (0, DEFAULT_FESTIVAL_DAYS)
    )

    estimates = []
    for airline, p in airlines.items():
        features = build_features(
            airline, source, destination, journey,
            p["dep_hour"], p["dep_min"], p["arr_hour"], p["arr_min"],
            p["duration_mins"], p["stops"],
            festival_flag, days_to_festival
        )
        price = float(model.predict(features)[0])
        estimates.append({
            "airline": airline,
            "typical_departure": f"{p['dep_hour']:02d}:{p['dep_min']:02d}",
            "typical_arrival": f"{p['arr_hour']:02d}:{p['arr_min']:02d}",
            "duration_mins": p["duration_mins"],
            "stops": p["stops"],
            "estimated_price": round(price, 2),
        })

    estimates.sort(key=lambda e: e["estimated_price"])
    prices = [e["estimated_price"] for e in estimates]

    return {
        "route": f"{source} -> {destination}",
        "date_of_journey": journey.isoformat(),
        "currency": "INR",
        "lowest_estimate": prices[0],
        "average_estimate": round(sum(prices) / len(prices), 2),
        "highest_estimate": prices[-1],
        "estimates_by_airline": estimates,
        "note": ("Estimates use the most common flight of each airline "
                 "on this route from the training data."),
    }


@app.post("/predict-detailed")
def predict_fare_detailed(flight: FlightInput):
    """ML-only estimate for a flight you describe in full."""
    dep = datetime.strptime(flight.dep_time, "%H:%M")
    arr = datetime.strptime(flight.arrival_time, "%H:%M")

    flight_data = build_features(
        flight.airline, flight.source, flight.destination,
        flight.date_of_journey,
        dep.hour, dep.minute, arr.hour, arr.minute,
        flight.duration_mins, STOPS_MAP[flight.total_stops],
        flight.festival_flag, flight.days_to_nearest_festival
    )

    try:
        predicted_price = float(model.predict(flight_data)[0])
    except Exception as error:
        raise HTTPException(status_code=400, detail=str(error))

    return {
        "predicted_price": round(predicted_price, 2),
        "currency": "INR"
    }


@app.post("/compare")
def compare_fares(request: CompareInput):
    """Live offers + ML estimate for each offer + comparison."""
    origin = CITY_TO_IATA.get(request.source)
    destination = CITY_TO_IATA.get(request.destination)

    if origin is None or destination is None:
        raise HTTPException(
            status_code=400,
            detail=f"Supported cities: {sorted(set(CITY_TO_IATA))}"
        )
    if origin == destination:
        raise HTTPException(
            status_code=400,
            detail="Source and destination must be different."
        )
    if request.date_of_journey < date.today():
        raise HTTPException(
            status_code=400,
            detail="Live search needs today's date or a future date."
        )

    offers = search_live_offers(
        origin, destination, request.date_of_journey,
        request.non_stop_only, request.max_offers
    )
    if not offers:
        raise HTTPException(
            status_code=404,
            detail="No live offers found for this route and date."
        )

    results = []
    for offer in offers:
        try:
            features, details = offer_to_features(
                offer, request.source, request.destination,
                request.festival_flag, request.days_to_nearest_festival
            )
            live_price = float(offer["price"]["grandTotal"])
            currency = offer["price"].get("currency", "INR")
            ml_price = float(model.predict(features)[0])
        except (KeyError, ValueError, IndexError):
            continue  # skip offers with unexpected structure

        entry = {
            **details,
            "live_price": round(live_price, 2),
            "ml_estimated_price": round(ml_price, 2),
            "currency": currency,
        }
        # Only compare prices when the live price is in INR
        if currency == "INR" and live_price > 0:
            entry["difference"] = round(ml_price - live_price, 2)
            entry["difference_percent"] = round(
                (ml_price - live_price) / live_price * 100, 2
            )
            entry["verdict"] = verdict(live_price, ml_price)
        results.append(entry)

    if not results:
        raise HTTPException(
            status_code=502,
            detail="Live offers could not be processed."
        )

    comparable = [r for r in results if "difference_percent" in r]
    summary = {}
    if comparable:
        cheapest = min(comparable, key=lambda r: r["live_price"])
        avg_live = sum(r["live_price"] for r in comparable) / len(comparable)
        avg_ml = sum(r["ml_estimated_price"] for r in comparable) / len(
            comparable
        )
        summary = {
            "cheapest_live_offer": cheapest,
            "average_live_price": round(avg_live, 2),
            "average_ml_estimate": round(avg_ml, 2),
            "average_difference_percent": round(
                (avg_ml - avg_live) / avg_live * 100, 2
            ),
        }

    return {
        "route": f"{origin} -> {destination}",
        "date_of_journey": request.date_of_journey.isoformat(),
        "offers_found": len(results),
        "summary": summary,
        "offers": results,
        "note": (
            "The ML model was trained on historical (2019) fares, so it "
            "can differ from today's prices. Offers from airlines not in "
            "the training data are less reliable."
        ),
    }
