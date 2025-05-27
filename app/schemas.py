from datetime import datetime
from typing import Optional

from fastapi import Query
from pydantic import BaseModel, Field


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
    time: datetime = Field(
        example="2022-11-12T13:14:15+00:00", ge=datetime(2011, 1, 1), le=datetime.now()
    )


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
    mcintosh_class: str = Field(title="McIntosh Classification", example="Bxo")
