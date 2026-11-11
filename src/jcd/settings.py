"""Runtime settings from environment variables / `.env`, plus file-based pipeline configs."""

from __future__ import annotations

from pathlib import Path

import yaml
from pydantic import BaseModel, ConfigDict, Field, SecretStr, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

from jcd.data.grouping import DEFAULT_PHASH_THRESHOLD


class Settings(BaseSettings):
    """Secrets and environment. Values are SecretStr so they never appear in reprs/logs."""

    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    roboflow_api_key: SecretStr | None = None
    hf_token: SecretStr | None = None


class SourceConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")
    workspace: str
    project: str
    version: int = Field(ge=1)
    format: str = "yolov8"


class SplitConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")
    ratios: dict[str, float] = {"train": 0.7, "valid": 0.15, "test": 0.15}
    seed: int = 42

    @field_validator("ratios")
    @classmethod
    def _known_splits(cls, v: dict[str, float]) -> dict[str, float]:
        unknown = set(v) - {"train", "valid", "test"}
        if unknown:
            raise ValueError(f"unknown split names: {sorted(unknown)}")
        return v


class HubConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")
    repo_id: str = Field(pattern=r"^[\w.-]+/[\w.-]+$")
    tag: str = Field(pattern=r"^v\d+\.\d+(\.\d+)?$")
    private: bool = False


class DataConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")
    source: SourceConfig
    raw_dir: Path
    processed_dir: Path
    split: SplitConfig = SplitConfig()
    hub: HubConfig | None = None
    phash_threshold: int = Field(DEFAULT_PHASH_THRESHOLD, ge=0, le=64)

    @classmethod
    def from_yaml(cls, path: str | Path) -> DataConfig:
        return cls.model_validate(yaml.safe_load(Path(path).read_text(encoding="utf-8")))
