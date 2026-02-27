from datetime import datetime, timedelta
from typing import List, Optional

from fastapi import Query
from pydantic import BaseModel, Field, model_validator

__all__ = [
    "ARCutoutClassificationInput",
    "McIntoshComponent",
    "McIntoshComponents",
    "ARCutoutClassificationResult",
    "ARDetectionInput",
    "HeliographicStonyhurstCoordinate",
    "BoundingBox",
    "ARDetection",
    "ARPTFlareForecast",
    "PTFlareForecast",
    "ARFlareProbability",
    "TSARFlareForecast",
    "TSFlareForecast",
]


class ARCutoutClassificationInput(BaseModel):
    time: datetime = Field(
        Query(title="Date time", ge=datetime(2011, 1, 1), le=datetime.now()),
        example="2022-11-12T13:14:15+00:00",
    )
    hgs_latitude: float = Field(
        Query(title="Heliographic Latitude", ge=-180, le=180, example=-70)
    )
    hgs_longitude: float = Field(
        Query(title="Heliographic Longitude", ge=-90, le=90, example=10)
    )


class McIntoshComponent(BaseModel):
    """Schema for individual McIntosh classification components (Z, p, c)"""

    predicted_class: str = Field(title="Predicted Class", example="B")
    probabilities: str = Field(
        title="Component Probabilities",
        example="A: 0.1000, B: 0.8000, C: 0.1000, H: 0.0500, LG: 0.0500",
    )


class McIntoshComponents(BaseModel):
    """Schema for all McIntosh classification components"""

    z: McIntoshComponent = Field(
        title="Z Component", description="Spot configuration (A, B, C, H, LG)"
    )
    p: McIntoshComponent = Field(
        title="p Component", description="Penumbra (asym, r, sym, x)"
    )
    c: McIntoshComponent = Field(
        title="c Component", description="Compactness (frag, o, x)"
    )


class ARCutoutClassificationResult(BaseModel):
    time: datetime = Field(
        title="Date time", ge=datetime(2011, 1, 1), le=datetime.now()
    )
    hgs_latitude: float = Field(
        title="Heliographic Latitude", ge=-180, le=180, example=-70
    )
    hgs_longitude: float = Field(
        title="Heliographic Longitude", ge=-90, le=90, example=10
    )
    hale_class: str = Field(title="Hale Classification", example="Beta")
    hale_probs: str = Field(
        title="Hale Probabilities",
        example="QS: 0.1000, IA: 0.2000, Alpha: 0.3000, Beta: 0.4000, Beta-Gamma: 0.0000",
    )
    mcintosh_class: str = Field(title="McIntosh Classification", example="Bxo")
    mcintosh_components: Optional[McIntoshComponents] = Field(
        title="McIntosh Components",
        description="Detailed breakdown of McIntosh classification components (only present for active regions)",
        default=None,
    )


class ARDetectionInput(BaseModel):
    time: datetime = Field(example="2022-11-12T13:14:15+00:00", ge=datetime(2011, 1, 1), le=datetime.now())


class HeliographicStonyhurstCoordinate(BaseModel):
    r"""
    Heliographic Stonyhurst (HGS) Coordinate
    """

    latitude: float = Field(title="Heliographic Latitude", ge=-180, le=180, example=-70)
    longitude: float = Field(title="Heliographic Longitude", ge=-90, le=90, example=10)


class BoundingBox(BaseModel):
    r"""
    Bounding Box
    """

    bottom_left: HeliographicStonyhurstCoordinate
    top_right: HeliographicStonyhurstCoordinate


class ARDetection(BaseModel):
    r"""
    Active Region Detection
    """

    time: datetime = Field(
        title="Date time", ge=datetime(2011, 1, 1), le=datetime.now()
    )
    bbox: BoundingBox
    hale_class: str = Field(title="Hale Classification", example="Beta")
    confidence: float = Field(title="Confidence", example="0.90")


class ARPTFlareForecast(BaseModel):
    noaa: int = Field(..., gt=0, description="Positive NOAA active region number")
    c: float = Field(..., ge=0.0, le=1.0, description="C-class flare probability")
    m: float = Field(..., ge=0.0, le=1.0, description="M-class flare probability")
    x: float = Field(..., ge=0.0, le=1.0, description="X-class flare probability")

    @model_validator(mode="after")
    def check_flare_hierarchy(self):
        if not (self.x <= self.m <= self.c):
            raise ValueError("Flare probabilities must satisfy: x <= m <= c")
        return self


class PTFlareForecast(BaseModel):
    timestamp: datetime = Field(..., description="Forecast timestamp (UTC)")
    forecasts: List[ARPTFlareForecast]


class ARFlareProbability(BaseModel):
    """Flare probabilities at a single point in the forecast horizon."""
    offset_minutes: int = Field(..., ge=0, description="Minutes from forecast timestamp")
    c: float = Field(..., ge=0.0, le=1.0, description="C-class flare probability")
    m: float = Field(..., ge=0.0, le=1.0, description="M-class flare probability")
    x: float = Field(..., ge=0.0, le=1.0, description="X-class flare probability")

    @model_validator(mode="after")
    def check_flare_hierarchy(self):
        if not (self.x <= self.m <= self.c):
            raise ValueError("Flare probabilities must satisfy: x <= m <= c")
        return self


class TSARFlareForecast(BaseModel):
    noaa: int = Field(..., gt=0, description="Positive NOAA active region number")
    probabilities: List[ARFlareProbability] = Field(
        ..., min_length=1, description="Time-ordered probability series"
    )

    @model_validator(mode="after")
    def check_offsets_ordered_and_unique(self):
        offsets = [p.offset_minutes for p in self.probabilities]
        if offsets != sorted(set(offsets)):
            raise ValueError("offset_minutes must be strictly increasing and unique")
        return self

    def at_offset(self, offset_minutes: int) -> ARFlareProbability | None:
        return next((p for p in self.probabilities if p.offset_minutes == offset_minutes), None)

    @property
    def horizon_minutes(self) -> int:
        return self.probabilities[-1].offset_minutes


class TSFlareForecast(BaseModel):
    timestamp: datetime = Field(..., description="Forecast timestamp (UTC)")
    step_minutes: int = Field(..., gt=0, description="Expected interval between steps (informational)")
    forecasts: List[TSARFlareForecast]

    @model_validator(mode="after")
    def check_consistent_offsets(self):
        if not self.forecasts:
            return self
        reference = [p.offset_minutes for p in self.forecasts[0].probabilities]
        for ar in self.forecasts[1:]:
            if [p.offset_minutes for p in ar.probabilities] != reference:
                raise ValueError(
                    f"AR {ar.noaa} has different time offsets than AR {self.forecasts[0].noaa}"
                )
        return self

    def absolute_times(self) -> List[datetime]:
        if not self.forecasts:
            return []
        return [
            self.timestamp + timedelta(minutes=p.offset_minutes)
            for p in self.forecasts[0].probabilities
        ]