from datetime import datetime

from app.schemas import ActiveRegionForecast, DailyFlareForecast, FlareForecast


def daily_flare_forecast(time: datetime) -> FlareForecast:
    mockforecast = FlareForecast(
        ars=[
            ActiveRegionForecast(
                timestamp=time,
                forecasts=[
                    DailyFlareForecast(
                        noaa=13664,
                        c=0.45,
                        m=0.25,
                        x=0.10,
                    ),
                    DailyFlareForecast(
                        noaa=13666,
                        c=0.50,
                        m=0.30,
                        x=0.15,
                    ),
                ],
            ),
            ActiveRegionForecast(
                timestamp=time,
                forecasts=[
                    DailyFlareForecast(
                        noaa=13654,
                        c=0.20,
                        m=0.08,
                        x=0.02,
                    )
                ],
            ),
        ]
    )
    return mockforecast
