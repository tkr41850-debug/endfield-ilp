"""
Raw resources
"""

from __future__ import annotations

from pathlib import Path
from typing import Final, TypeAlias

import numpy as np
import yaml

with open(Path(__file__).resolve().parent / "items.yaml", "r") as file:
    _data: Final[dict] = yaml.safe_load(file.read())
    raw_resources: Final[list[str]] = list(_data["raw_resources"])
N: Final = len(raw_resources)
POWER: Final = raw_resources.index("power")

T: TypeAlias = np.ndarray


class ResourceCost:
    def __init__(self, val: T = np.zeros((N,), np.uint16)) -> None:
        self.val: Final = val

    @classmethod
    def from_dict(cls, d: dict) -> ResourceCost:
        val = np.zeros((N,), np.uint16)
        for i, k in enumerate(raw_resources):
            val[i] = d.get(k, 0)
        return ResourceCost(val)

    def __repr__(self) -> str:
        return ", ".join(
            [
                (
                    f"{raw_resources[i]} @ :red[**{self.val[i]}**]/min"
                    if raw_resources[i] != "power"
                    else f":yellow[**{self.val[i]}**]W"
                )
                for i in range(N)
                if self.val[i]
            ]
        )

    def __add__(self, other: ResourceCost) -> ResourceCost:
        return ResourceCost(self.val + other.val)

    def __mul__(self, k: int) -> ResourceCost:
        return ResourceCost(self.val * k)
