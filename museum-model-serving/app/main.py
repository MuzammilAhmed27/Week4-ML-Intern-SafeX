import json
import logging
import time
import uuid
from pathlib import Path

import joblib
import pandas as pd
from fastapi import FastAPI, Request
from pydantic import BaseModel, ConfigDict, Field

# Project paths
BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_DIR = BASE_DIR / "models"

# Logging: do not log prediction inputs or sensitive payloads
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s %(message)s",
)
logger = logging.getLogger("museum_api")

app = FastAPI(
    title="Museum Model Serving API",
    description="Versioned inference API using synthetic museum-object data.",
    version="1.0.0",
)


def load_model(version: str):
    """Load one trained model and its metadata."""
    version_dir = MODEL_DIR / version
    model_path = version_dir / "model.joblib"
    metadata_path = version_dir / "metadata.json"

    if not model_path.is_file():
        raise FileNotFoundError(f"Model file not found: {model_path}")

    if not metadata_path.is_file():
        raise FileNotFoundError(f"Metadata file not found: {metadata_path}")

    model = joblib.load(model_path)

    with metadata_path.open("r", encoding="utf-8") as file:
        metadata = json.load(file)

    return model, metadata


# Load both versions at application startup/import.
MODELS = {
    "v1": load_model("v1"),
    "v2": load_model("v2"),
}


class MuseumObject(BaseModel):
    """Input features expected by the trained models."""

    model_config = ConfigDict(extra="forbid")

    age_years: float = Field(ge=0, le=5000)
    size_cm: float = Field(gt=0, le=250)
    estimated_value_usd: float = Field(ge=10, le=50000)

    organic: int = Field(strict=True, ge=0, le=1)
    has_inscription: int = Field(strict=True, ge=0, le=1)
    excavation_find: int = Field(strict=True, ge=0, le=1)
    natural_specimen: int = Field(strict=True, ge=0, le=1)
    artwork: int = Field(strict=True, ge=0, le=1)


@app.middleware("http")
async def log_requests(request: Request, call_next):
    request_id = str(uuid.uuid4())
    request.state.request_id = request_id
    start_time = time.perf_counter()

    try:
        response = await call_next(request)
    except Exception:
        duration_ms = round((time.perf_counter() - start_time) * 1000, 2)
        logger.exception(
            "request_failed request_id=%s method=%s path=%s duration_ms=%s",
            request_id,
            request.method,
            request.url.path,
            duration_ms,
        )
        raise

    duration_ms = round((time.perf_counter() - start_time) * 1000, 2)
    response.headers["X-Request-ID"] = request_id

    # Only log request metadata, never the submitted features.
    logger.info(
        "request_complete request_id=%s method=%s path=%s status=%s duration_ms=%s",
        request_id,
        request.method,
        request.url.path,
        response.status_code,
        duration_ms,
    )
    return response


def make_prediction(data: MuseumObject, version: str, request_id: str):
    model, metadata = MODELS[version]

    default_features = [
        "age_years",
        "size_cm",
        "estimated_value_usd",
        "organic",
        "has_inscription",
        "excavation_find",
        "natural_specimen",
        "artwork",
    ]

    features = metadata.get("features", default_features)
    values = data.model_dump()

    # Preserve the feature order used during training.
    input_data = pd.DataFrame(
        [[values[feature] for feature in features]],
        columns=features,
    )

    prediction = str(model.predict(input_data)[0])

    result = {
        "request_id": request_id,
        "model_version": version,
        "prediction": prediction,
    }

    if hasattr(model, "predict_proba"):
        probabilities = model.predict_proba(input_data)[0]

        classes = getattr(model, "classes_", None)
        if classes is None and hasattr(model, "named_steps"):
            final_model = list(model.named_steps.values())[-1]
            classes = getattr(final_model, "classes_", None)

        if classes is not None:
            result["probabilities"] = {
                str(label): round(float(probability), 4)
                for label, probability in zip(classes, probabilities)
            }

    return result


@app.get("/")
def root():
    return {
        "service": "Museum Model Serving API",
        "documentation": "/docs",
    }


@app.get("/health")
def health():
    return {
        "status": "ok",
        "models": {
            version: {
                "loaded": model is not None,
                "metadata_loaded": metadata is not None,
            }
            for version, (model, metadata) in MODELS.items()
        },
    }


@app.get("/model-info")
def model_info_v1():
    return {
        "model_version": "v1",
        "metadata": MODELS["v1"][1],
    }


@app.get("/v2/model-info")
def model_info_v2():
    return {
        "model_version": "v2",
        "metadata": MODELS["v2"][1],
    }


@app.post("/predict")
def predict_v1(data: MuseumObject, request: Request):
    return make_prediction(
        data,
        "v1",
        getattr(request.state, "request_id", "unknown"),
    )


@app.post("/v2/predict")
def predict_v2(data: MuseumObject, request: Request):
    return make_prediction(
        data,
        "v2",
        getattr(request.state, "request_id", "unknown"),
    )
