from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from battery_management.assets.battery import Battery
from battery_management.battery_optimizer import create

plt.style.use("ggplot")
np.random.seed(42)

# Define Sinus shaped Prices (this is quite accurate for a 24h cycle)
x = np.linspace(0, 2 * np.pi, 30)
prices_import = np.sin(x)

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
        "connected": [True] * 30,
        "capacity": 40,
        "efficiency_charge": 0.8,
        "efficiency_discharge": 0.8,
    }
)

battery2 = Battery(
    **{
        "id": 23,
        "capacity": 40,
        "energy_start": 12,
        "energy_end": 40,
        "energy_min": 5,
        "energy_max": 40,
        "power_charge_max": 5,
        "power_discharge_max": 5,
        "connected": [False] * 5 + [True] * 25,
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

# First Optimization: Prices only
results = fo.optimize()

# Second Optimization: Prices and Marketed Volumes
#  And here's the kicker: we can use pos/neg
#  Don't go too low, otherwise you hit the lower floor
marketed_volumes1 = np.array([-0.5] * 5 + [np.nan] * 25)
fo.add_marketed_volumes(marketed_volumes1)
results1 = fo.optimize()

# Third Optimization: Prices and Updated Marketed Volumes
marketed_volumes2 = np.array([4] * 3 + [np.nan] * 27)
fo.add_marketed_volumes(marketed_volumes2)
results2 = fo.optimize()

# -- Plotting --
plt.figure(figsize=(24, 24))
for i, (res, prices, marketed_volumes) in enumerate(
    zip(
        [results, results1, results2],
        [prices_import] * 3,
        [None, marketed_volumes1, marketed_volumes2],
    )
):
    bat_res = res.battery_results
    df = res.aggregated_results
    plt.subplot(4, 3, 1 + i)
    plt.plot(bat_res.loc[23]["power_kw"], label="Power 23")
    plt.plot(bat_res.loc[42]["power_kw"], label="Power 42")
    plt.xlabel("Time")
    plt.ylabel("Power (Battery) [kW]")
    plt.legend()

    plt.subplot(4, 3, 4 + i)
    plt.plot(df["power_kw_site"], label="Power All")
    if marketed_volumes is not None:
        plt.plot(marketed_volumes, label="Marketed Volumes")
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
    plt.plot(prices_import)
    plt.xlabel("Time")
    plt.ylabel("Prices [pound/kWh]")

file_name = Path(__file__).parents[2] / Path("figures/basic_optimization.png")
file_name.parents[0].mkdir(parents=True, exist_ok=True)
plt.savefig(file_name)
