import os
from pathlib import Path

from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    data_path: Path = Path(os.getenv("DATAPATH", "/arccnet/data"))
    model_path: Path = Path(os.getenv("MODESLPATH", "/arccnet/models"))


settings = Settings()
