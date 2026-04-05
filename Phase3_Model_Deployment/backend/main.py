import json
from pathlib import Path
from typing import List

import joblib
import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel, Field, field_validator


BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_DIR = BASE_DIR / "models"
PIPELINE_PATH = MODEL_DIR / "chicago_crime_pipeline.pkl"
LABELS_PATH = MODEL_DIR / "label_classes.json"
METADATA_PATH = MODEL_DIR / "model_metadata.json"


DAY_NAME_TO_INT = {
    "Monday": 0,
    "Tuesday": 1,
    "Wednesday": 2,
    "Thursday": 3,
    "Friday": 4,
    "Saturday": 5,
    "Sunday": 6,
}


class PredictionInput(BaseModel):
    hour: int = Field(..., ge=0, le=23)
    day_of_week: int | str = Field(..., description="0-6 or Monday-Sunday")
    month: int = Field(..., ge=1, le=12)
    is_weekend: int | bool = Field(..., description="0/1 or boolean")
    community_area_id: int = Field(..., ge=1, le=77)
    distance_to_nearest_station: float = Field(..., ge=0)
    stations_within_500m: int = Field(..., ge=0)
    community_type: str = Field(..., min_length=1)

    @field_validator("day_of_week", mode="before")
    @classmethod
    def normalize_day_of_week(cls, value):
        if isinstance(value, str):
            value = value.strip()
            if value.isdigit():
                value = int(value)
            else:
                if value not in DAY_NAME_TO_INT:
                    raise ValueError("day_of_week must be 0-6 or a valid weekday name")
                value = DAY_NAME_TO_INT[value]
        if not isinstance(value, int) or not (0 <= value <= 6):
            raise ValueError("day_of_week must be an integer from 0 to 6")
        return value

    @field_validator("is_weekend", mode="before")
    @classmethod
    def normalize_is_weekend(cls, value):
        if isinstance(value, bool):
            return int(value)
        if isinstance(value, str):
            value = value.strip().lower()
            if value in {"yes", "true", "1"}:
                return 1
            if value in {"no", "false", "0"}:
                return 0
        if isinstance(value, int) and value in {0, 1}:
            return value
        raise ValueError("is_weekend must be 0/1, true/false, or yes/no")

    @field_validator("community_type")
    @classmethod
    def normalize_community_type(cls, value: str):
        value = value.strip()
        if not value:
            raise ValueError("community_type cannot be empty")
        return value


class TopPrediction(BaseModel):
    label: str
    probability: float


class PredictionResponse(BaseModel):
    predicted_label: str
    predicted_probability: float
    top_3_predictions: List[TopPrediction]
    model_version: str


app = FastAPI(title="Chicago Crime Prediction API")

pipeline = None
class_labels = None
metadata = None
feature_columns = None


@app.on_event("startup")
def load_assets() -> None:
    global pipeline, class_labels, metadata, feature_columns

    pipeline = joblib.load(PIPELINE_PATH)

    with open(LABELS_PATH, "r", encoding="utf-8") as f:
        class_labels = json.load(f)

    with open(METADATA_PATH, "r", encoding="utf-8") as f:
        metadata = json.load(f)

    feature_columns = metadata["feature_columns"]


@app.get("/")
def root():
    return {"status": "ok", "service": "Chicago Crime Prediction API"}


@app.get("/health")
def health():
    return {
        "status": "healthy",
        "model_loaded": pipeline is not None,
        "n_classes": len(class_labels) if class_labels else 0,
    }


@app.get("/model-info")
def model_info():
    return {
        "model_type": metadata.get("model_type", "unknown"),
        "target_column": metadata.get("target_column", "crime_category"),
        "feature_columns": metadata.get("feature_columns", []),
        "class_labels": class_labels,
        "metrics": metadata.get("metrics", {}),
    }


@app.post("/predict", response_model=PredictionResponse)
def predict(payload: PredictionInput):
    row = {
        "hour": payload.hour,
        "day_of_week": payload.day_of_week,
        "month": payload.month,
        "is_weekend": payload.is_weekend,
        "community_area_id": payload.community_area_id,
        "distance_to_nearest_station": payload.distance_to_nearest_station,
        "stations_within_500m": payload.stations_within_500m,
        "community_type": payload.community_type,
    }

    X = pd.DataFrame([row])[feature_columns]
    probabilities = pipeline.predict_proba(X)[0]

    top_idx = probabilities.argsort()[::-1][:3]
    top_3_predictions = [
        {
            "label": class_labels[i],
            "probability": float(probabilities[i]),
        }
        for i in top_idx
    ]

    return PredictionResponse(
        predicted_label=top_3_predictions[0]["label"],
        predicted_probability=top_3_predictions[0]["probability"],
        top_3_predictions=top_3_predictions,
        model_version=metadata.get("model_type", "LightGBM multiclass pipeline"),
    )
