from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from battery_management.assets.stationary_battery import StationaryBattery
from battery_management.battery_optimizer import create

np.random.seed(42)
plt.style.use("ggplot")
df = pd.read_csv(f"{Path(__file__).parent}/data/optimization_testcase.csv")
prices = df["price_buy"].values
pv = df["pv"].values
charging = df["load"].values
site_load = charging - pv

time_steps = len(df)

# ------------ Optimizer --------------
# Define two batteries along the V2G parameters

# Determine cycle cost. The battery has a lifetime of 5000 cycles and costs 35 000 Euro. A full cycle corresponds to
# charging and discharging from 0.09 * 210 to 210 * 0.99 (i.e. 0.9 * 210 kWh) or 189kWh. So one cycle is 378 kWh and
# costs 7 Euro. The resulting penalty here is (7/378) Eur/kWh = 0.0185 Eur/kWh applied to the absolute value of any
# energy changes.

battery = StationaryBattery(
    **{
        "id": 42,
        "capacity": 210,
        "energy_start": 0.8202 * 210,
        "energy_end": 0.1 * 210,
        "energy_min": 0.1 * 210,
        "energy_max": 210,
        "power_charge_min": 0,
        "power_charge_max": 350,
        "power_discharge_max": 350,
        "connected": [True] * time_steps,
        "efficiency_charge": 1,
        "efficiency_discharge": 1,
        "cycle_life": 5000,
        "cycle_usage": 0,
        "battery_costs": 35000,
    }
)

fo = create(id=42, type="OR", dt=0.25)

# Activate Necessary Flags
fo.calculate_savings = False
fo.allow_curtailment = False
fo.include_battery_costs = True

fo.add_battery(battery)
fo.add_prices(prices, prices)

fo.add_site_load(site_load)
fo.add_site_limits(site_load_restriction_discharge=0)

# Optimization
price_flex_pos = np.array([1] * time_steps)
fo.add_flex(
    prices_flex_pos=price_flex_pos,
    prices_flex_neg=price_flex_pos,
    symmetrical_flex=True,
)
prl_marketed_volumes = np.zeros(time_steps)
fo.add_marketed_flex(pos_flex=prl_marketed_volumes, neg_flex=prl_marketed_volumes)
results = fo.optimize()

# -- Plotting --
n_col = 1
n_row = 5
plt.figure(figsize=(24, 24))
for i, (title, res, prices, prices_pos, prices_neg) in enumerate(
    zip(
        [
            "PRL",
        ],
        [
            results,
        ],
        [
            prices * 0 + 1,
        ],
        [
            price_flex_pos,
        ],
        [
            price_flex_pos,
        ],
    )
):
    bat_res = res.battery_results
    site_res = res.site_results

    marketed_volumes = None
    plt.subplot(n_row, n_col, 1 + i)
    for bat in res.batteries:
        plt.plot(bat_res.loc[bat.id]["power_kw"], label=f"Power {bat.id}")

        if "flex_pos" in bat_res.columns:
            # Positive flexibility means reducing the power (charging less up to discharging fully)
            plt.fill_between(
                bat_res.loc[bat.id].index.values,
                bat_res.loc[bat.id]["power_kw"] - bat_res.loc[bat.id]["flex_pos"],
                bat_res.loc[bat.id]["power_kw"],
                alpha=0.2,
            )
        if "flex_neg" in bat_res.columns:
            # Positive flexibility means reducing the power (charging less up to discharging fully)
            plt.fill_between(
                bat_res.loc[bat.id].index.values,
                bat_res.loc[bat.id]["power_kw"],
                bat_res.loc[bat.id]["power_kw"] + bat_res.loc[bat.id]["flex_neg"],
                alpha=0.2,
            )
    plt.xlabel("Time")
    plt.ylabel("Power (Battery) [kW]")
    plt.legend()
    plt.title(title, fontsize=20)

    plt.subplot(n_row, n_col, n_col + 1 + i)
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

    plt.subplot(n_row, n_col, 2 * n_col + 1 + i)
    for bat in results.batteries:
        plt.plot(bat_res.loc[bat.id]["energy_content_kwh"], label=f"Power {bat.id}")

    plt.xlabel("Time")
    plt.ylabel("Energy Content [kWh]")
    plt.legend()

    plt.subplot(n_row, n_col, 3 * n_col + 1 + i)
    plt.plot(prices, label="DA")
    if prices_pos is not None:
        plt.plot(prices_pos, label="Pos. Flex")
    if prices_neg is not None:
        plt.plot(prices_neg, label="Neg. Flex")
    plt.xlabel("Time")
    plt.ylabel("Prices [pound/kWh]")
    plt.legend()

    plt.subplot(n_row, n_col, 4 * n_col + 1 + i)
    plt.plot(site_load, label="Site Load")
    plt.plot(pv, label="PV")
    plt.plot(charging, label="Demand")
    plt.xlabel("Time")
    plt.ylabel("Power [kW]")
    plt.legend()

    plt.suptitle("Different Scenarios of Flexibility Optimization", fontsize=26)

file_name = Path(__file__).parents[2] / Path("figures/flexev_prl.png")
file_name.parents[0].mkdir(parents=True, exist_ok=True)
plt.savefig(file_name)
