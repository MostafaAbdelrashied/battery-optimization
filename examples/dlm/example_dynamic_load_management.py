"""
Example of Dynamic Load Management, optimizing load on 3 charging points to reduce peak load.

"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from battery_management.assets.battery import Battery
from battery_management.assets.charging_point import ChargingPoint
from battery_management.optimizer.battery_optimization_or import FleetOptimizationOR

np.random.seed(42)
plt.style.use("ggplot")
# Define number of timesteps
n_t = 30

# Define Time discretization
dt = 0.25  # hours so 15min granularity

# Define prices
prices_import = [
    0.30,
] * n_t  # Euro/kWh
capacity_cost = 30  # Euro/kW

# Define Site Limit
limit_purchase = 40

# Define Sinus shaped Site Load
site_load_index = pd.date_range(
    start=pd.Timestamp(2020, 2, 1), freq="15min", periods=n_t
)
# Initialize real batteries using the forecast
site_load = pd.DataFrame(index=site_load_index)

x = np.linspace(0, 2 * np.pi, n_t)
site_load["power_kw"] = np.abs(np.cos(x) * 15)


# Define 3 batteries (partly asynchronous)
battery_efficiency = 0.9
battery1 = Battery(
    **{
        "id": 42,
        "capacity": 40,
        "energy_start": 10,
        "energy_end": 40,
        "energy_min": 5,
        "energy_max": 40,
        "power_charge_max": 11,
        "power_discharge_max": 0,
        "connected": [True] * 20 + [False] * 10,
        "efficiency_charge": battery_efficiency,
        "efficiency_discharge": battery_efficiency,
    }
)

battery2 = Battery(
    **{
        "id": 23,
        "capacity": 30,
        "energy_start": 12,
        "energy_end": 28,
        "energy_min": 5,
        "energy_max": 28,
        "power_charge_max": 11,
        "power_discharge_max": 0,
        "connected": [False] * 21 + [True] * 9,
        "efficiency_charge": battery_efficiency,
        "efficiency_discharge": battery_efficiency,
    }
)

battery3 = Battery(
    **{
        "id": 24,
        "capacity": 50,
        "energy_start": 12,
        "energy_end": 48,
        "energy_min": 5,
        "energy_max": 48,
        "power_charge_max": 11,
        "power_discharge_max": 0,
        "connected": [False] * 15 + [True] * 15,
        "efficiency_charge": battery_efficiency,
        "efficiency_discharge": battery_efficiency,
    }
)

batteries = [battery1, battery2, battery3]

# Define 3 Charging points
charging_point_power = 11  # kW
charging_points = [
    ChargingPoint(dict(asset_id=i + 1, charging_power_kw=charging_point_power))
    for i in range(3)
]

# Instantiate the Optimizer
# optimizer type: OR Tools
# we use dt = 0.5 <=> 1 time step is 30min
# - Add batteries
# - Add Prices


fo = FleetOptimizationOR(
    dt=dt,
    fully_charged_as_penalty=True,
    single_continuous_session_allowed=False,
    penalize_spiky_behaviour=True,
)
_ = [fo.add_battery(battery) for battery in batteries]

fo.add_date_range(site_load_index)
_ = [fo.add_charging_point(cp) for cp in charging_points]

fo.add_prices(tariffs_import=prices_import, capacity_tariffs_import=capacity_cost)

fo.add_site_load(site_load=site_load["power_kw"].values)

fo.add_site_limits(site_load_restriction_charge=limit_purchase, limit_as_penalty=True)

results = fo.optimize()
res = results.aggregated_results
bat_res = results.battery_results

# -- Plotting --
plt.figure(figsize=(24, 24))
plt.subplot(2, 2, 1)
for bat in batteries:
    plt.plot(bat_res.loc[bat.id]["power_kw"], ":", label=f"Power {bat.id}")

plt.xlabel("Time")
plt.ylabel("Power (Battery) [kW]")
plt.legend()

plt.subplot(2, 2, 2)
plt.plot(res["power_kw_site"] / dt, label="CP Load")
plt.plot(res["power_kw_site"] / dt + site_load["power_kw"], label="Total Load")
plt.plot(site_load["power_kw"], label="Site Load")
plt.axhline(limit_purchase, label="Site Limit")
# TODO Repair
# plt.axhline(results.site_constraint_purchase, label="New Site Limit", c="k")
plt.xlabel("Time")
plt.ylabel("Power (Aggregated) [kW]")
plt.legend()

plt.subplot(2, 2, 3)
for bat in batteries:
    plt.plot(bat_res.loc[bat.id]["energy_content_kwh"], label=f"Battery {bat.id}")

plt.xlabel("Time")
plt.ylabel("Energy [kWh]")
plt.legend()

plt.subplot(2, 2, 4)
plt.plot(prices_import)
plt.xlabel("Time")
plt.ylabel("Prices [pound/kWh]")

file_name = Path(__file__).parents[2] / Path("figures/dlm_example.png")
file_name.parents[0].mkdir(parents=True, exist_ok=True)
plt.savefig(file_name)
