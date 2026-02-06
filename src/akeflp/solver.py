from typing import MutableSequence, NamedTuple, Optional, Sequence, Tuple, cast

import numpy as np
from scipy.optimize import linprog

from akef.items import items, power_sources
from akef.resource import POWER, ResourceCost, raw_resources


class TaskWithPower(NamedTuple):
    x: int
    """how many instances are running at all times"""

    power: int
    """how much power this allocation generates"""

    opportunity_cost: int
    """how much value/min you sacrifice"""


class TaskWithValue(NamedTuple):
    x: int
    """how many instances of this facility are running"""

    rate: int
    """how much value/min you gain as a result"""


class SolveResult(NamedTuple):
    power_required: int
    power_total: int
    power: dict[str, TaskWithPower]
    produce: dict[str, TaskWithValue]
    value_rate: int


class TaskDetail(NamedTuple):
    value: int
    """value per item"""

    lower_bound: int
    """minimum instances of this"""

    upper_bound: int
    """maximum instances of this"""


def solve(
    constraints: ResourceCost,
    tasks: dict[str, TaskDetail],
    max_rate: int = 8000 // (60 * 24),
    disallowed_taints: list[str] = [],
) -> SolveResult:
    # things that you can make
    xlabels = list(
        set([k for k, v in tasks.items() if v] + list(power_sources.keys()))
        - set(raw_resources)
    )
    N = len(xlabels)

    # what will you use for power? in terms of how many units are active?
    plabels = list(power_sources.keys())
    K = len(plabels)
    c = np.hstack(
        [
            # note this is a maximization problem so coefficients are inverted
            # since scipy.optimize.linprog only does minimization problems
            np.asarray(
                [
                    # using it as power!!
                    (
                        tasks[k].value * power_sources[k].consumption_rate
                        if k in tasks
                        else 0
                    )
                    for k in plabels
                ]
            ),
            np.asarray(
                [
                    # value per minute, per instance
                    -(tasks[i].value * items[i].output_rate if i in tasks else 0)
                    for i in xlabels
                ]
            ),
        ]
    )

    A: list[Sequence[float | int]] = []
    b: list[float] = []

    # make sure power is at least baselinepower
    A.append(
        [-power_sources[k].power_output for k in plabels]
        + [items[k].cost.val[POWER] for k in xlabels]
    )
    b.append(int(constraints.val[POWER]))

    # make sure power_source usage is non-negative (>= 0)
    for i, plabel in enumerate(plabels):
        Aps_tmp = [0.0] * (N + K)
        Aps_tmp[i] = power_sources[plabel].consumption_rate  # consuming it
        if plabel in xlabels:
            Aps_tmp[K + xlabels.index(plabel)] = -items[plabel].output_rate  # making it
        # for batteries (sane resources), you don't just *get* them
        # however, originium ore is a mined ore so...
        # i guess its kinda complicated
        if plabel not in raw_resources:
            A.append(Aps_tmp)
            b.append(0)

    # make sure net item output does not exceed max_rate
    for i, xlabel in enumerate(xlabels):
        Axs_tmp = [0.0] * (N + K)
        Axs_tmp[K + i] = items[xlabel].output_rate  # creating it
        if xlabel in plabels:  # some output might be allocated for power
            Axs_tmp[plabels.index(xlabel)] = -power_sources[xlabel].consumption_rate
        if xlabel not in raw_resources:
            A.append(Axs_tmp)
            b.append(max_rate)

    # make sure general resource usage is no more than constraints <=
    for i, k in enumerate(raw_resources):
        if k == "power":
            continue
        Ag_tmp = [0.0] * K + [items[k].cost.val[i] for k in xlabels]
        if k in plabels:
            # special case for originium_ore which is both a power source and a resource
            Ag_tmp[plabels.index(k)] = power_sources[k].consumption_rate
        A.append(Ag_tmp)
        b.append(int(constraints.val[i]))

    A_ub = np.asarray(A)
    b_ub = np.asarray(b)
    bounds: Sequence[Tuple[Optional[int], Optional[int]]] = [
        (0, None) for _ in range(K)
    ] + [
        (
            (tasks[xlabels[i]].lower_bound, tasks[xlabels[i]].upper_bound)
            if xlabels[i] in tasks
            else (0, None)
        )
        for i in range(N)
    ]

    # ban items that have disallowed_taints
    for i, xlabel in enumerate(xlabels):
        for bad in disallowed_taints:
            if bad in items[xlabel].taints:
                cast(MutableSequence, bounds)[i + K] = (0, 0)

    # print(plabels, xlabels)
    # print(A_ub, b_ub)

    res = linprog(c, A_ub=A_ub, b_ub=b_ub, bounds=bounds, integrality=1)
    # print(res)

    return SolveResult(
        value_rate=-round(res.fun),
        power_total=round(
            res.x[:K] @ np.asarray([x.power_output for x in power_sources.values()])
            + constraints.val[POWER]
        ),
        power_required=round(
            res.x[K : K + N]
            @ np.asarray([int(items[k].cost.val[POWER]) for k in xlabels])
        ),
        power={
            k: TaskWithPower(
                x=round(res.x[i]),
                power=round(res.x[i]) * power_sources[k].power_output,
                opportunity_cost=round(res.x[i]) * c[i],
            )
            for i, k in enumerate(plabels)
            if res.x[i] > 0
        },
        produce={
            k: TaskWithValue(
                x=round(res.x[i + K]),
                rate=-round(res.x[i + K] * c[i + K]),
            )
            for i, k in enumerate(xlabels)
            if round(res.x[i + K])
        },
    )


# max cx subject to Ax <= b
# c: Unknown,
# A_ub: Unknown | None = None,
# b_ub: Unknown | None = None,

if __name__ == "__main__":
    # smoke test for infinite power, make sure solver finds basic solution
    # constrained by upper_bound (if unbounded, origocrust 18-belts)
    res = solve(
        constraints=ResourceCost.from_dict({"power": 1000, "originium_ore": 540}),
        tasks={
            # "ferrium": TaskDetail(value=5, lower_bound=0, upper_bound=100),
            # "ferrium_part": TaskDetail(value=15, lower_bound=0, upper_bound=100),
            # "originium_ore": TaskDetail(value=1, lower_bound=0, upper_bound=100),
            "origocrust": TaskDetail(value=10, lower_bound=0, upper_bound=10),
            # "lc_valley_battery": TaskDetail(value=1, lower_bound=0, upper_bound=100),
            # "sc_valley_battery": TaskDetail(value=1, lower_bound=0, upper_bound=100),
            # "hc_valley_battery": TaskDetail(value=1, lower_bound=0, upper_bound=100),
        },
        max_rate=100000,
    )
    print(res)
