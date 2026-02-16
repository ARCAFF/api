from datetime import datetime
from typing import List

from app.schemas import FlareForecast, ARFlareForecast


def daily_flare_forecast(time: datetime) -> FlareForecast:
    mock_forecast = FlareForecast(
        timestamp=time,
        forecasts=[
            ARFlareForecast(
                noaa=13664,
                c=0.45,
                m=0.25,
                x=0.10,
            ),
            ARFlareForecast(
                noaa=13666,
                c=0.50,
                m=0.30,
                x=0.15,
            ),
        ],
    )

    return mock_forecast
