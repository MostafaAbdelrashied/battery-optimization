from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from battery_management.assets.battery import Battery
from battery_management.assets.stationary_battery import StationaryBattery
from battery_management.battery_optimizer import create

np.random.seed(42)
plt.style.use("ggplot")
# =====================================================================
# Arguments to try:
symmetrical_flex = True

# =====================================================================

# Define Sinus shaped Prices (this is quite accurate for a 24h cycle)
n_t = 30
x = np.linspace(0, 2 * np.pi, n_t)
prices_import = x * 0

# Define two batteries along the V2G parameters
battery1 = Battery(
    **{
        "id": 42,
        "energy_start": 10,
        "energy_end": 40,
        "energy_min": 5,
        "energy_max": 40,
        "power_charge_max": 5,
        "power_discharge_max": 5,
        "connected": [True] * n_t,
        "capacity": 40,
        "efficiency_charge": 0.8,
        "efficiency_discharge": 0.8,
    }
)

battery2 = StationaryBattery(
    **{
        "id": 23,
        "capacity": 40,
        "energy_start": 12,
        "energy_end": 40,
        "energy_min": 5,
        "energy_max": 40,
        "power_charge_max": 5,
        "power_discharge_max": 5,
        "connected": [False] * 5 + [True] * (n_t - 5),
        "efficiency_charge": 0.8,
        "efficiency_discharge": 0.8,
    }
)

# Instantiate the Optimizer
# optimizer type: OR Tools
# we use dt = 0.5 <=> 1 time step is 30min
# - Add batteries
# - Add Prices

fo = create(id=42, dt=0.5, type="OR")
_ = [fo.add_battery(battery) for battery in [battery1, battery2]]
fo.add_prices(prices_import, prices_import)

# First Optimization: Same prices for pos and neg - static
price_flex_pos = np.array([1] * 30)
fo.add_flex(
    prices_flex_pos=price_flex_pos,
    prices_flex_neg=price_flex_pos,
    symmetrical_flex=symmetrical_flex,
)
results = fo.optimize()

# Second Optimization: Same prices for pos and neg - dynamic
price_flex_pos1 = np.array([0] * 13 + [1] * 4 + [0] * 13)
fo.add_flex(
    prices_flex_pos=price_flex_pos1,
    prices_flex_neg=price_flex_pos1,
    symmetrical_flex=symmetrical_flex,
)
results1 = fo.optimize()

# Third Optimization: Prices and Updated Marketed Volumes
price_flex_neg = np.array([0] * 18 + [1] * 4 + [0] * 8)
fo.add_flex(
    prices_flex_pos=price_flex_pos1,
    prices_flex_neg=price_flex_neg,
    symmetrical_flex=symmetrical_flex,
)
results2 = fo.optimize()

# -- Plotting --
plt.figure(figsize=(24, 24))
for i, (title, res, prices, prices_pos, prices_neg) in enumerate(
    zip(
        [
            "Static Prices",
            "Dynamic Prices",
            "Different Prices Pos/Neg",
        ],
        [results, results1, results2],
        [prices_import * 0 + 1, prices_import * 0 + 1, prices_import],
        [price_flex_pos, price_flex_pos1, price_flex_pos1],
        [price_flex_pos, price_flex_pos1, price_flex_neg],
    )
):
    marketed_volumes = None
    bat_res = res.battery_results
    site_res = res.site_results
    print("=" * 50)
    print(f"{title}:")
    print("=" * 50)
    print(site_res)
    plt.subplot(4, 3, 1 + i)
    for bat in res.batteries:
        _res = bat_res.loc[bat.id]
        plt.plot(_res["power_kw"], label=f"Power {bat.id}")

        if "flex_pos" in bat_res.columns:
            # Positive flexibility means reducing the power (charging less up to discharging fully)
            plt.fill_between(
                _res.index.values,
                _res["power_kw"] - _res["flex_pos"],
                _res["power_kw"],
                alpha=0.2,
            )
        if "flex_neg" in bat_res.columns:
            # Positive flexibility means reducing the power (charging less up to discharging fully)
            plt.fill_between(
                _res.index.values,
                _res["power_kw"],
                _res["power_kw"] + _res["flex_neg"],
                alpha=0.2,
            )
    plt.xlabel("Time")
    plt.ylabel("Power (Battery) [kW]")
    plt.legend()
    plt.title(title, fontsize=20)

    plt.subplot(4, 3, 4 + i)
    plt.plot(site_res["power_kw"], label="Power All")
    if marketed_volumes is not None:
        plt.plot(marketed_volumes, label="Marketed Volumes")
    if "FlexPos" in site_res.columns:
        # Positive flexibility means reducing the power (charging less up to discharging fully)
        plt.fill_between(
            site_res.index.values,
            site_res["power_kw"] - site_res["FlexPos"],
            site_res["power_kw"],
            alpha=0.2,
        )

    if "FlexNeg" in site_res.columns:
        # Positive flexibility means reducing the power (charging less up to discharging fully)
        plt.fill_between(
            site_res.index.values,
            site_res["power_kw"],
            site_res["power_kw"] + site_res["FlexNeg"],
            alpha=0.2,
        )

    plt.xlabel("Time")
    plt.ylabel("Power (Aggregated) [kW]")
    plt.legend()

    plt.subplot(4, 3, 7 + i)
    plt.plot(bat_res.loc[23]["energy_content_kwh"], label="Battery 23")
    plt.plot(bat_res.loc[42]["energy_content_kwh"], label="Battery 42")
    plt.xlabel("Time")
    plt.ylabel("Energy Content [kWh]")
    plt.legend()

    plt.subplot(4, 3, 10 + i)
    plt.plot(prices, label="DA")
    if prices_pos is not None:
        plt.plot(prices_pos, label="Pos. Flex")
    if prices_neg is not None:
        plt.plot(prices_neg, label="Neg. Flex")
    plt.xlabel("Time")
    plt.ylabel("Prices [pound/kWh]")
    plt.legend()
    plt.suptitle("Different Scenarios of Flexibility Optimization", fontsize=26)

file_name = Path(__file__).parents[2] / Path("figures/flexev_flexibility_posneg_2.png")
file_name.parents[0].mkdir(parents=True, exist_ok=True)
plt.savefig(file_name)
