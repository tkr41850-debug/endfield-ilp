import math

import streamlit as st

from akef.items import Item, items, raw_resources


def render(item: Item, rate: float, depth: int = 0) -> None:
    totcost = item.cost * math.ceil(rate / item.base_rate)
    rateinfo = f"{item.name} @ {rate}/min costs {totcost}"
    if not item.inputs:
        st.write(rateinfo)
        return
    with st.expander(rateinfo):
        st.write(f"to {item.action} costs {item.action_overhead}")
        for amt, pitem in item.inputs:
            recipe_rate = rate / item.output
            render(pitem, amt * recipe_rate, depth + 1)


def main() -> None:
    for item in items.values():
        if item.name in raw_resources:
            continue
        render(item, item.base_rate * item.output)
