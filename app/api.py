from typing import List

from fastapi import APIRouter, Depends

from app.classify import classify, detect
from app.forecast import daily_flare_forecast
from app.schemas import (ARCutoutClassificationInput,
                         ARCutoutClassificationResult, ARDetection,
                         ARDetectionInput, FlareForecast)

classificaiton_router = APIRouter()
forecast_router = APIRouter()


@classificaiton_router.post(
    "/arcnet/classify_cutout/", tags=["AR Cutout Classification"]
)
async def classify_cutout(
    classification_request: ARCutoutClassificationInput,
) -> ARCutoutClassificationResult:
    r"""
    Classify an AR cutout generated from a magnetogram at the given date and location as json data.
    """
    classification = classify(
        time=classification_request.time,
        hgs_latitude=classification_request.hgs_latitude,
        hgs_longitude=classification_request.hgs_longitude,
    )
    classification_result = ARCutoutClassificationResult.model_validate(classification)
    return classification_result


@classificaiton_router.get(
    "/arcnet/full_disk_detection", tags=["Full disk AR Detection"]
)
async def full_disk_detection(
    detection_request: ARDetectionInput = Depends(),
) -> List[ARDetection]:
    r"""
    Detect and classify all ARs in a magnetogram at the given date as a URL parameter.
    """
    detections = detect(detection_request.time)
    detection_result = [ARDetection.model_validate(d) for d in detections]
    return detection_result


@forecast_router.post("/flare_forecast", tags=["Flare Forecast"])
async def flare_forecast(detection_request: ARDetectionInput) -> FlareForecast:
    r"""
    Flare forecast for next 24 hours
    """
    forecast_result = daily_flare_forecast(detection_request.time)
    return forecast_result
