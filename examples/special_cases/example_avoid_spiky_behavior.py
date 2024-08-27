from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from battery_management.assets.battery import Battery
from battery_management.optimizer.battery_optimization_or import FleetOptimizationOR

np.random.seed(42)
plt.style.use("ggplot")


def main():
    n_steps = 20
    # Define Sinus shaped Prices (this is quite accurate for a 24h cycle)
    x = np.linspace(0, n_steps - 1, n_steps)
    prices = np.where(x < 14, 0.1, 1)
    noise = 1 + np.random.randn(n_steps) / 1000
    prices *= noise

    # Define two batteries along the V2G parameters
    battery1 = Battery(
        **{
            "id": 42,
            "energy_start": 10,
            "energy_end": 40,
            "energy_min": 5,
            "energy_max": 50,
            "capacity": 50,
            "power_charge_max": 10,
            "power_discharge_max": 10,
            "connected": [True] * n_steps,
            "efficiency_charge": 0.95,
            "efficiency_discharge": 0.95,
        }
    )

    battery2 = Battery(
        **{
            "id": 23,
            "energy_start": 12,
            "energy_end": 40,
            "energy_min": 5,
            "energy_max": 40,
            "capacity": 40,
            "power_charge_max": 10,
            "power_discharge_max": 10,
            "connected": [True] * n_steps,
            "efficiency_charge": 0.8,
            "efficiency_discharge": 0.8,
        }
    )

    # Instantiate the Optimizer
    # we use dt = 0.5 <=> 1 time step is 30min
    # - Add batteries
    # - Add Prices
    fo = FleetOptimizationOR(id=42, dt=0.5, fully_charged_as_penalty=True)
    _ = [fo.add_battery(battery) for battery in [battery1, battery2]]
    fo.add_prices(prices, prices)

    # First Optimization: Prices only
    res = fo.optimize()
    fo.plot(res, f"{Path(__file__).parents[2]}/figures/spiky_behaviour.png")

    fo = FleetOptimizationOR(
        id=42, dt=0.5, penalize_spiky_behaviour=True, fully_charged_as_penalty=True
    )
    _ = [fo.add_battery(battery) for battery in [battery1, battery2]]
    fo.add_prices(prices, prices)

    # First Optimization: Prices only
    res = fo.optimize()
    fo.plot(res, f"{Path(__file__).parents[2]}/figures/not_spiky_behaviour.png")


if __name__ == "__main__":
    main()
