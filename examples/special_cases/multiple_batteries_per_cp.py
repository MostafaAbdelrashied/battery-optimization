from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from battery_management.assets.battery import Battery
from battery_management.assets.charging_point import ChargingPoint
from battery_management.battery_optimizer import create

np.random.seed(42)
plt.style.use("ggplot")

# Define Sinus shaped Prices (this is quite accurate for a 24h cycle)
x = np.linspace(0, 2 * np.pi, 30)
prices_import = np.sin(x)
date_range = pd.date_range(start=pd.Timestamp(2020, 4, 1), freq="30min", periods=30)

# Define two batteries along the V2G parameters
day_sessions = [
    Battery(
        **{
            "id": 1 + i,
            "capacity": 40,
            "energy_start": 10 + i,
            "energy_end": 40 - i,
            "energy_min": 5,
            "energy_max": 40,
            "power_charge_max": 5,
            "power_discharge_max": 5,
            "connected": [True] * 10 + [False] * 20,
            "efficiency_charge": 0.8,
            "efficiency_discharge": 0.8,
        }
    )
    for i in range(2)
]

night_sessions = [
    Battery(
        **{
            "id": 100 + i,
            "capacity": 40,
            "energy_start": 32 - i,
            "energy_end": 38 + i,
            "energy_min": 5,
            "energy_max": 40,
            "power_charge_max": 10,
            "power_discharge_max": 10,
            "connected": [False] * 15 + [True] * 15,
            "efficiency_charge": 0.8,
            "efficiency_discharge": 0.8,
        }
    )
    for i in range(3)
]

batteries = day_sessions + night_sessions

charging_points = [
    ChargingPoint(
        dict(
            asset_id=42 + i,
            charging_power_kw=10 * (i + 1),
            discharging_power_kw=10 * (i + 1),
        )
    )
    for i in range(2)
]

# Instantiate the Optimizer
# we use dt = 0.5 <=> 1 time step is 30min
# - Add batteries
# - Add Prices
fo = create(id=42, type="OR", dt=0.5, fully_charged_as_penalty=True)
_ = [fo.add_battery(battery) for battery in batteries]
fo.add_date_range(date_range)
_ = [fo.add_charging_point(cp) for cp in charging_points]
fo.add_prices(prices_import, prices_import)

# First Optimization: Prices only
res = fo.optimize()

# Second Optimization: Prices and Marketed Volumes
#  And here's the kicker: we can use pos/neg
#  Don't go too low, otherwise you hit the lower floor
marketed_volumes1 = np.array([-0.5] * 5 + [np.nan] * 25)
fo.add_marketed_volumes(marketed_volumes1)
res1 = fo.optimize()

# Third Optimization: Prices and Updated Marketed Volumes
marketed_volumes2 = np.array([4] * 3 + [np.nan] * 27)
fo.add_marketed_volumes(marketed_volumes2)
res2 = fo.optimize()


plt.figure(figsize=(24, 24))
for i, (res, prices, marketed_volumes) in enumerate(
    zip(
        [res, res1, res2],
        [prices_import] * 3,
        [None, marketed_volumes1, marketed_volumes2],
    )
):
    bat_res = res.battery_results
    site_res = res.site_results

    n_rows = 5
    n_col = 3
    plt.subplot(n_rows, n_col, 1 + i)
    for bat in batteries:
        if bat.id in bat_res.index:
            plt.plot(bat_res.loc[bat.id]["power_kw"], label=f"Power {bat.id}")

    plt.xlabel("Time")
    plt.ylabel("Power (Battery) [kW]")
    plt.legend()

    plt.subplot(n_rows, n_col, n_col + 1 + i)
    plt.plot(site_res["power_kw"], label="Delta Energy")
    if marketed_volumes is not None:
        plt.plot(site_res["power_kw"].index, marketed_volumes, label="Marketed Volumes")
    plt.xlabel("Time")
    plt.ylabel("Power (Aggregated) [kW]")
    plt.legend()

    plt.subplot(n_rows, n_col, n_col * 2 + 1 + i)
    for bat in batteries:
        if bat.id in bat_res.index:
            plt.plot(
                bat_res.loc[bat.id]["energy_content_kwh"], label=f"Battery {bat.id}"
            )

    plt.xlabel("Time")
    plt.ylabel("Energy [kWh]")
    plt.legend()

    plt.subplot(n_rows, n_col, n_col * 3 + 1 + i)
    plt.plot(prices_import)
    plt.xlabel("Time")
    plt.ylabel("Prices [pound/kWh]")

    plt.subplot(n_rows, n_col, n_col * 4 + 1 + i)
    for bat in batteries:
        if bat.id in bat_res.index:
            plt.plot(bat.connected, label=bat.id)
    plt.legend()
    plt.xlabel("Time")
    plt.ylabel("Connected")


file_name = Path(__file__).parents[2] / Path("figures/Multiple_Batteries_per_CP.png")
file_name.parents[0].mkdir(parents=True, exist_ok=True)
plt.savefig(file_name)
