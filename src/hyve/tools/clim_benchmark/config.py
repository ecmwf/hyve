# SPDX-FileCopyrightText: 2026 European Centre for Medium-Range Weather Forecasts (ECMWF)
#
# SPDX-License-Identifier: Apache-2.0

"""Configuration schema for the clim-benchmark tool."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

Stride = Literal["daily", "weekly", "monthly"]

# Minimum window size (in days) required for each stride category.
STRIDE_MIN_WINDOW: dict[str, int] = {
    "daily": 1,
    "weekly": 7,
    "monthly": 30,
}

# Size (in days) of the step between sampled days inside the centered window.
STRIDE_DAYS: dict[str, int] = {
    "daily": 1,
    "weekly": 7,
    "monthly": 30,
}


class ClimConfig(BaseModel):
    """User-facing configuration for the climatology computation.

    All fields are validated at construction; `validate_against_data`
    performs the additional check that depends on the inferred dataset
    timestep.
    """

    model_config = ConfigDict(extra="forbid")

    window_days: int = Field(
        default=31,
        description="Total width of the centered window in days. Must be odd and >= stride minimum.",
    )
    stride: Stride = Field(
        default="daily",
        description="Sampling stride inside the window.",
    )
    issue_frequency_hours: int = Field(
        default=24,
        description="Hours between successive forecast issues. Must divide 24 and be >= dataset timestep.",
    )
    percentiles: list[int] = Field(
        default_factory=lambda: list(range(0, 101, 10)),
        description="Percentile values in [0, 100] to compute.",
    )
    start_date: str | None = None
    end_date: str | None = None
    num_workers: int = Field(
        default=1,
        description="Number of concurrent workers used for percentile computation.",
    )

    @field_validator("window_days")
    @classmethod
    def _window_positive_odd(cls, v: int) -> int:
        if v <= 0:
            raise ValueError(f"window_days must be positive, got {v}")
        if v % 2 == 0:
            raise ValueError(
                f"window_days must be odd for unambiguous centering, got {v}"
            )
        return v

    @field_validator("issue_frequency_hours")
    @classmethod
    def _issue_freq_divides_24(cls, v: int) -> int:
        if v <= 0 or v > 24 or 24 % v != 0:
            raise ValueError(
                f"issue_frequency_hours must be a positive divisor of 24, got {v}"
            )
        return v

    @field_validator("percentiles")
    @classmethod
    def _percentiles_valid(cls, v: list[int]) -> list[int]:
        if len(v) == 0:
            raise ValueError("percentiles must be non-empty")
        if any(p < 0 or p > 100 for p in v):
            raise ValueError(f"percentiles must be in [0, 100], got {v}")
        if len(set(v)) != len(v):
            raise ValueError(f"percentiles must be unique, got {v}")
        return sorted(v)

    @field_validator("num_workers")
    @classmethod
    def _num_workers_valid(cls, v: int) -> int:
        if v < 1:
            raise ValueError(f"num_workers must be >= 1, got {v}")
        return v

    @model_validator(mode="after")
    def _window_vs_stride(self) -> "ClimConfig":
        min_window = STRIDE_MIN_WINDOW[self.stride]
        if self.window_days < min_window:
            raise ValueError(
                f"stride={self.stride!r} requires window_days >= {min_window}, "
                f"got window_days={self.window_days}"
            )
        return self

    def validate_against_data(self, timestep_hours: float) -> None:
        """Check config consistency with the dataset's inferred timestep."""
        if timestep_hours <= 0:
            raise ValueError(f"timestep_hours must be positive, got {timestep_hours}")
        if timestep_hours > 24:
            raise ValueError(
                f"timestep_hours={timestep_hours} exceeds 24; sub-daily "
                "or daily data required"
            )
        # issue_frequency_hours must be an integer multiple of timestep
        ratio = self.issue_frequency_hours / timestep_hours
        if not float(ratio).is_integer():
            raise ValueError(
                f"issue_frequency_hours={self.issue_frequency_hours} is not a "
                f"multiple of the dataset timestep ({timestep_hours}h)"
            )
