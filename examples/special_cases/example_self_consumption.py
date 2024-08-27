from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from battery_management.assets.battery import Battery
from battery_management.optimizer.battery_optimization_or import FleetOptimizationOR

# ------------ Inputs --------------
x = np.linspace(0, 2, 48)
pv = np.cos(x * 2 * np.pi) - 0.25
pv = np.where(pv > 0, pv * 15, 0)
charging = (np.sin(x * 3 * np.pi) + 1) / 2 + 2

site_load = charging - pv
time_steps = len(pv)

tariff_import = np.ones(time_steps)
tariff_export = np.zeros(time_steps)
battery_capacity_kwh = 20


# ------------ Optimizer --------------

# Define two batteries along the V2G parameters
battery = Battery(
    **{
        "id": 42,
        "capacity": battery_capacity_kwh,
        "energy_start": 0.8 * battery_capacity_kwh,
        "energy_end": battery_capacity_kwh * 0.5,
        "energy_min": battery_capacity_kwh * 0.2,
        "energy_max": battery_capacity_kwh * 0.8,
        "power_charge_max": 350,
        "power_discharge_max": 350,
        "connected": [True] * time_steps,
        "efficiency_charge": 0.95,
        "efficiency_discharge": 0.95,
        "stationary": True,
    }
)

fo = FleetOptimizationOR(id=42, dt=0.25, fully_charged_as_penalty=False)
fo.calculate_savings = False
fo.allow_curtailment = True
fo.include_site_load_costs = False

fo.add_battery(battery)
fo.add_prices(tariffs_import=tariff_import, tariffs_export=tariff_export)

fo.add_site_load(site_load)

# First Optimization: Prices only
res = fo.optimize()
result = res.aggregated_results

print(result.head(5))

print(res.objective_value)

# ------------ Plotting ------------
plt.figure(figsize=(24, 8))

# 1) Compare Energy
plt.subplot(2, 2, 1)
result["energy_content_kwh"].plot(label="General Optimizer")
plt.legend(title="SOC [kWh]")

plt.subplot(2, 2, 2)
plt.plot(tariff_import, label="Import Price")
plt.plot(tariff_export, label="Export Price")
plt.legend()

plt.subplot(2, 2, 3)
plt.plot(pv, label="Photovoltaic")
plt.plot(charging, label="Site Load")
plt.plot(result["power_kw_site"], label="Battery Load")
plt.plot(
    site_load + result["power_kw_site"], label="Effective Site Load"
)  # TO TAKE FROM OPTIMIZER
plt.legend()

plt.subplot(2, 2, 4)
plt.legend()

plt.savefig(f"{Path(__file__).parents[2]}/figures/self_consumption.png")
