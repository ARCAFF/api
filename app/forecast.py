from datetime import datetime

from app.schemas import FlareForecast, DailyFlareForecast, ActiveRegionForecast


def daily_flare_forecast(time: datetime) -> FlareForecast:
    mockforecast = FlareForecast(
        ars=[
            ActiveRegionForecast(
                noaa=13664,
                forecasts=[
                    DailyFlareForecast(
                        timestamp=time,
                        c=0.45,
                        m=0.25,
                        x=0.10,
                    ),
                    DailyFlareForecast(
                        timestamp=time,
                        c=0.50,
                        m=0.30,
                        x=0.15,
                    ),
                ],
            ),
            ActiveRegionForecast(
                noaa=13665,
                forecasts=[
                    DailyFlareForecast(
                        timestamp=time,
                        c=0.20,
                        m=0.08,
                        x=0.02,
                    )
                ],
            ),
        ]
    )
    return mockforecast