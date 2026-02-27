from datetime import datetime, timezone
from typing import List

from app.schemas import PTFlareForecast, ARPTFlareForecast, ARFlareProbability, TSFlareForecast, TSARFlareForecast


def pt_daily_flare_forecast(time: datetime) -> PTFlareForecast:
    mock_forecast = PTFlareForecast(
        timestamp=time,
        forecasts=[
            ARPTFlareForecast(
                noaa=13664,
                c=0.45,
                m=0.25,
                x=0.10,
            ),
            ARPTFlareForecast(
                noaa=13666,
                c=0.50,
                m=0.30,
                x=0.15,
            ),
        ],
    )

    return mock_forecast

def ts_flare_forecast(time: datetime) -> TSFlareForecast:
    # Single AR forecast — high activity, 1-hour steps over 24 hours
    mock_forecast = TSFlareForecast(
        timestamp=datetime(2024, 11, 1, 6, 0, 0, tzinfo=timezone.utc),
        step_minutes=60,
        forecasts=[
            TSARFlareForecast(
                noaa=13490,
                probabilities=[
                    ARFlareProbability(
                        offset_minutes=offset,
                        c=round(0.80 - i * 0.01, 3),
                        m=round(0.40 - i * 0.005, 3),
                        x=round(0.08 - i * 0.001, 3),
                    )
                    for i, offset in enumerate(range(0, 25 * 60, 60))  # 0, 60, 120, ... 1440
                ],
            )
        ],
    )

    return mock_forecast