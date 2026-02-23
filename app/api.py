from typing import List

from fastapi import APIRouter, Depends

from app.classify import classify, detect
from app.forecast import daily_flare_forecast
from app.schemas import (
    ARCutoutClassificationInput,
    ARCutoutClassificationResult,
    ARDetection,
    ARDetectionInput,
    FlareForecast,
)

classification_router = APIRouter()
forecast_router = APIRouter()


def _perform_classification(
    request: ARCutoutClassificationInput,
) -> ARCutoutClassificationResult:
    classification = classify(
        time=request.time,
        hgs_latitude=request.hgs_latitude,
        hgs_longitude=request.hgs_longitude,
    )
    return ARCutoutClassificationResult.model_validate(classification)


@classification_router.get(
    "/arcnet/classify_cutout/", tags=["AR Cutout Classification"]
)
async def classify_cutout_get(
    request: ARCutoutClassificationInput = Depends(),
) -> ARCutoutClassificationResult:
    r"""
    Classify an AR cutout generated from a magnetogram at the given date and location as JSON data.
    """
    return _perform_classification(request)


@classification_router.post(
    "/arcnet/classify_cutout/", tags=["AR Cutout Classification"]
)
async def classify_cutout_post(
    request: ARCutoutClassificationInput,
) -> ARCutoutClassificationResult:
    r"""
    Classify an AR cutout generated from a magnetogram at the given date and location as JSON data.
    """
    return _perform_classification(request)


def _perform_detection(request: ARDetectionInput) -> List[ARDetection]:
    detections = detect(request.time)
    return [ARDetection.model_validate(d) for d in detections]


@classification_router.get(
    "/arcnet/full_disk_detection", tags=["Full disk AR Detection"]
)
async def full_disk_detection_get(
    request: ARDetectionInput = Depends(),
) -> List[ARDetection]:
    r"""
    Detect and classify all ARs in a magnetogram at the given date as a URL parameter.
    """
    return _perform_detection(request)


@classification_router.post(
    "/arcnet/full_disk_detection", tags=["Full disk AR Detection"]
)
async def full_disk_detection_post(
    request: ARDetectionInput,
) -> List[ARDetection]:
    r"""
    Detect and classify all ARs in a magnetogram at the given date as a URL parameter.
    """
    return _perform_detection(request)


@forecast_router.get("/flare_forecast", tags=["Flare Forecast"])
async def flare_forecast_get(
    request: ARDetectionInput = Depends(),
) -> FlareForecast:
    r"""
    Flare forecast for next 24 hours
    """
    forecast_result = daily_flare_forecast(request.time)
    return forecast_result


@forecast_router.post("/flare_forecast", tags=["Flare Forecast"])
async def flare_forecast_post(request: ARDetectionInput) -> FlareForecast:
    r"""
    Flare forecast for next 24 hours
    """
    forecast_result = daily_flare_forecast(request.time)
    return forecast_result
