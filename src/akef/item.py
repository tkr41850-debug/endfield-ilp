from __future__ import annotations

import math
from typing import Final, Sequence, Tuple, TypeAlias

from akef.resource import ResourceCost

Rate: TypeAlias = int  # items per minute


class Item:
    def __init__(
        self,
        name: str,
        seconds_to_craft: int,
        overhead: ResourceCost,
        inputs: Sequence[Tuple[int, Item]],
        action: str,
        output: int = 1,
        value: int = 1,
    ) -> None:
        self.name: Final = name
        self.base_rate: Final = 60 / seconds_to_craft
        self.cost: Final = sum(
            [
                item.cost
                * math.ceil(w * self.base_rate / (item.base_rate * item.output))
                for w, item in inputs
            ],
            overhead,
        )
        self.output: Final = output
        self.inputs: Final = inputs
        self.action: Final = action
        self.action_overhead: Final = overhead
        self.value: Final = value
