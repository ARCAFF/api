import time

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware

from app.api import classificaiton_router, forecast_router

main_descripion = """
# ARCAFF API
Active Region Classification and Flare Forecasting (ARCAFF) API
"""

app = FastAPI(
    title="ARCAFF",
    description=main_descripion,
    contact={
        "name": "ARCAFF",
        "url": "http://www.arcaff.eu",
    },
)


app.add_middleware(
    CORSMiddleware,
    allow_origins="*",
    allow_credentials=True,
    allow_methods=["get", "post", "options"],
    allow_headers=["*"],
)


@app.middleware("http")
async def add_process_time_header(request: Request, call_next):
    start_time = time.time()
    response = await call_next(request)
    process_time = time.time() - start_time
    response.headers["X-Process-Time"] = str(process_time)
    return response


classification_description = """
### ARCNET

Active Region Classification Network (ARCNET) provides an API to AR classifications.
#### AR Cutout Classification

Given a AR magnetogram cutout return the classifications Hale (magnetic) and McInstosh (modified Zurich) classifications.

#### AR Detection

Given a full disk magnetogram return the bounding boxes and classification for each detected AR.

"""


classification_tags_metadata = [
    {
        "name": "AR Cutout Classification",
        "description": "Classify cutouts generated from a magnetogram at the given date and location.",
    },
    {
        "name": "Full disk AR Detection",
        "description": "Detect and classify all AR from a magnetogram for the given date.",
    },
]

classifcation_app = FastAPI(
    title="ARCNET",
    description=classification_description,
    summary="Active Region Classification and Flare Forecasting (ARCAFF) API",
    version="0.0.1",
    contact={
        "name": "ARCAFF",
        "url": "http://www.arcaff.eu",
    },
    openapi_tags=classification_tags_metadata,
)
classifcation_app.include_router(classificaiton_router)
app.mount("/classification", classifcation_app)

flares_desciption = """
#### Flare forecast

Flare forecasts.

"""

flares_tags_metadata = [
    {
        "name": "Forecast",
        "description": "Forecast solar flares.",
    },
]

forecast_app = FastAPI(
    title="Flares",
    description=flares_desciption,
    summary="Active Region Classification and Flare Forecasting (ARCAFF) API",
    version="0.0.1",
    contact={
        "name": "ARCAFF",
        "url": "http://www.arcaff.eu",
    },
    openapi_tags=flares_tags_metadata,
)
forecast_app.include_router(forecast_router)

app.mount("/forecast", forecast_app)
