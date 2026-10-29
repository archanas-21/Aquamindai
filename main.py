from fastapi import FastAPI, HTTPException

from fastapi.middleware.cors import (
    CORSMiddleware
)

from services.data_service import (
    load_data,
    latest_station_data
)

from services.anomaly_service import (
    detect_anomalies
)

from services.risk_service import (
    apply_risk_scores
)


# ==================================================
# APPLICATION
# ==================================================

app = FastAPI(
    title="AquaMind AI API",
    description=(
        "AI-Powered Groundwater "
        "Intelligence and Early Warning Platform"
    ),
    version="1.0.0"
)


# ==================================================
# CORS
# ==================================================

app.add_middleware(
    CORSMiddleware,

    allow_origins=[
        "http://localhost:5173"
    ],

    allow_credentials=True,

    allow_methods=[
        "*"
    ],

    allow_headers=[
        "*"
    ],
)


# ==================================================
# PROCESS DATA
# ==================================================

def get_processed_data():

    df = load_data()

    df = detect_anomalies(
        df
    )

    df = apply_risk_scores(
        df
    )

    return df


# ==================================================
# ROOT
# ==================================================

@app.get("/")
def root():

    return {
        "project": "AquaMind AI",
        "status": "running",
        "message": (
            "Groundwater Intelligence API"
        )
    }


# ==================================================
# HEALTH CHECK
# ==================================================

@app.get("/api/health")
def health():

    df = load_data()

    return {
        "status": "healthy",

        "records": int(
            len(df)
        ),

        "stations": int(
            df["Station"].nunique()
        ),

        "districts": int(
            df["District"].nunique()
        )
    }


# ==================================================
# DASHBOARD
# ==================================================

@app.get("/api/dashboard")
def dashboard():

    df = get_processed_data()

    latest = (
        df
        .sort_values("Date_Time")
        .groupby(
            "Station",
            as_index=False
        )
        .last()
    )

    if latest.empty:

        return {
            "stations": 0,
            "districts": 0,
            "average_level": 0,
            "alerts": 0,
            "average_risk": 0,
            "highest_risk": 0,
            "highest_risk_district": "N/A"
        }

    district_risk = (
        latest
        .groupby("District")[
            "Risk Score"
        ]
        .mean()
        .sort_values(
            ascending=False
        )
    )

    highest_risk_district = (
        district_risk.index[0]
        if not district_risk.empty
        else "N/A"
    )

    return {

        "stations": int(
            latest["Station"].nunique()
        ),

        "districts": int(
            latest["District"].nunique()
        ),

        "average_level": round(
            float(
                latest[
                    "Groundwater_Level"
                ].mean()
            ),
            2
        ),

        "alerts": int(
            (
                latest["Status"]
                == "Abnormal"
            ).sum()
        ),

        "average_risk": round(
            float(
                latest["Risk Score"].mean()
            ),
            1
        ),

        "highest_risk": int(
            latest["Risk Score"].max()
        ),

        "highest_risk_district":
            highest_risk_district
    }


# ==================================================
# STATIONS
# ==================================================

@app.get("/api/stations")
def stations():

    df = get_processed_data()

    latest = (
        df
        .sort_values("Date_Time")
        .groupby(
            "Station",
            as_index=False
        )
        .last()
    )

    records = latest.to_dict(
        orient="records"
    )

    for record in records:

        record["Date_Time"] = (
            record["Date_Time"]
            .isoformat()
        )

    return records


# ==================================================
# DISTRICTS
# ==================================================

@app.get("/api/districts")
def districts():

    df = get_processed_data()

    result = (
        df
        .groupby("District")
        .agg(
            Average_Level=(
                "Groundwater_Level",
                "mean"
            ),

            Average_Risk=(
                "Risk Score",
                "mean"
            ),

            Stations=(
                "Station",
                "nunique"
            )
        )
        .reset_index()
    )

    records = result.to_dict(
        orient="records"
    )

    for record in records:

        record["Average_Level"] = round(
            float(
                record["Average_Level"]
            ),
            2
        )

        record["Average_Risk"] = round(
            float(
                record["Average_Risk"]
            ),
            1
        )

        record["Stations"] = int(
            record["Stations"]
        )

    return records


# ==================================================
# ALERTS
# ==================================================

@app.get("/api/alerts")
def alerts():

    df = get_processed_data()

    result = df[
        df["Status"] == "Abnormal"
    ].copy()

    result = result.sort_values(
        "Date_Time",
        ascending=False
    )

    result = result.head(100)

    records = result.to_dict(
        orient="records"
    )

    for record in records:

        record["Date_Time"] = (
            record["Date_Time"]
            .isoformat()
        )

    return records


# ==================================================
# RISK
# ==================================================

@app.get("/api/risk")
def risk():

    df = get_processed_data()

    result = (
        df
        .groupby("District")
        .agg(
            Risk_Score=(
                "Risk Score",
                "mean"
            ),

            Stations=(
                "Station",
                "nunique"
            )
        )
        .reset_index()
        .sort_values(
            "Risk_Score",
            ascending=False
        )
    )

    records = result.to_dict(
        orient="records"
    )

    for record in records:

        record["Risk_Score"] = round(
            float(
                record["Risk_Score"]
            ),
            1
        )

        record["Stations"] = int(
            record["Stations"]
        )

    return records


# ==================================================
# SINGLE STATION
# ==================================================

@app.get(
    "/api/station/{station_name}"
)
def station_details(
    station_name: str
):

    df = get_processed_data()

    result = df[
        df["Station"]
        == station_name
    ]

    if result.empty:

        raise HTTPException(
            status_code=404,
            detail="Station not found"
        )

    latest = (
        result
        .sort_values("Date_Time")
        .iloc[-1]
    )

    history = result[
        [
            "Date_Time",
            "Groundwater_Level",
            "Risk Score",
            "Status"
        ]
    ].copy()

    history_records = (
        history.to_dict(
            orient="records"
        )
    )

    for record in history_records:

        record["Date_Time"] = (
            record["Date_Time"]
            .isoformat()
        )

        record[
            "Groundwater_Level"
        ] = float(
            record[
                "Groundwater_Level"
            ]
        )

        record[
            "Risk Score"
        ] = int(
            record["Risk Score"]
        )

    return {

        "station": latest["Station"],

        "district": latest["District"],

        "state": latest["State"],

        "latitude": float(
            latest["Latitude"]
        ),

        "longitude": float(
            latest["Longitude"]
        ),

        "groundwater_level": float(
            latest["Groundwater_Level"]
        ),

        "status": latest["Status"],

        "risk_score": int(
            latest["Risk Score"]
        ),

        "risk_level": latest["Risk Level"],

        "possible_cause":
            latest["Possible Cause"],

        "recommendation":
            latest["Recommendation"],

        "history": history_records
    }