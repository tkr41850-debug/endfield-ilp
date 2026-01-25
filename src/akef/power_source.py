from __future__ import annotations

from typing import Final


class PowerSource:
    def __init__(self, duration_seconds: int, power_output: int) -> None:
        self.duration_seconds: Final = duration_seconds
        self.power_output: Final = power_output

    @classmethod
    def from_dict(cls, d: dict) -> PowerSource:
        return PowerSource(duration_seconds=d["seconds"], power_output=d["power"])
