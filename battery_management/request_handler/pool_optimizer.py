from typing import List, Optional

import numpy as np
import pandas as pd
from loguru import logger

from battery_management.assets.battery import Battery
from battery_management.helper.pool_result import PoolResult
from battery_management.optimizer.battery_optimization_or import FleetOptimizationOR


class PoolOptimizer:
    def __init__(
        self,
        fleet_optimizer: List[FleetOptimizationOR],
        marketed_volumes: Optional[np.ndarray] = None,
        datetime_index: Optional[pd.DatetimeIndex] = None,
    ):
        """
        Initialize the PoolOptimizer.

        Parameters
        ----------
        fleet_optimizer : List[FleetOptimizationOR]
            List of fleet optimizers.
        marketed_volumes : Optional[np.ndarray], default=None
            Marketed volumes to be matched.
        datetime_index : Optional[pd.DatetimeIndex], default=None
            Date-time index for the results.
        """
        self.fleet_optimizer = fleet_optimizer
        self.marketed_volumes = marketed_volumes
        self.datetime_index = datetime_index

    def optimize(self, type: str = "simple") -> PoolResult:
        """
        Perform pool optimization.

        Parameters
        ----------
        type : str, default="simple"
            Type of optimization to use.

        Returns
        -------
        PoolResult
            Result of the pool optimization.
        """
        logger.info("-" * 125)
        logger.info("Start Pool Optimization")
        logger.info("-" * 125)
        if self.datetime_index is not None:
            logger.info(
                f"Period: {min(self.datetime_index).isoformat()}--{max(self.datetime_index).isoformat()}, "
                f"{len(self.datetime_index)} steps"
            )

        if (self.marketed_volumes is None) or (len(self.fleet_optimizer) < 2):
            result = self._disconnected()
        else:
            result = self._simple_dispatcher()

        return result

    def _simple_dispatcher(self, fractions: Optional[np.ndarray] = None) -> PoolResult:
        """
        Perform simple dispatcher optimization.

        Parameters
        ----------
        fractions : Optional[np.ndarray], default=None
            Fractions for distributing marketed volumes.

        Returns
        -------
        PoolResult
            Result of the simple dispatcher optimization.
        """
        start_optimization = pd.Timestamp.now()
        logger.info("Starting Simple Dispatcher Optimization")

        if fractions is None:
            fractions = np.array([len(fo.batteries) for fo in self.fleet_optimizer])
            fractions = fractions / np.sum(fractions)

        result = {}
        for fraction, fo in zip(fractions, self.fleet_optimizer):
            fo.add_marketed_volumes(self.marketed_volumes * fraction)
            fleet_result = fo.optimize()
            result[fo.id] = fleet_result

        time_elapsed = (pd.Timestamp.now() - start_optimization).total_seconds()
        logger.info(f"Time elapsed for Pool Optimization: {time_elapsed}s")

        return PoolResult(result)

    def _disconnected(self) -> PoolResult:
        """
        Perform disconnected optimization.

        Returns
        -------
        PoolResult
            Result of the disconnected optimization.
        """
        start_optimization = pd.Timestamp.now()
        logger.info("Starting Disconnected Optimization")

        result = {}
        for fo in self.fleet_optimizer:
            logger.info(f"Optimize Fleet {fo.id}")
            if self.marketed_volumes is not None:
                fo.add_marketed_volumes(self.marketed_volumes)
            fleet_result = fo.optimize()
            logger.info(
                f"Objective Value for Fleet {fo.id}: {fleet_result.objective_value}"
            )
            if fleet_result.success == 0:
                logger.debug("Optimization successful")
            else:
                logger.warning(f"Fleet optimization {fo.id} failed!")
            result[fo.id] = fleet_result

        time_elapsed = (pd.Timestamp.now() - start_optimization).total_seconds()
        logger.info(f"Time elapsed for disconnected Pool Optimization: {time_elapsed}s")

        return PoolResult(result)

    def __len__(self) -> int:
        return len(self.fleet_optimizer)


if __name__ == "__main__":
    x = np.linspace(0, 2 * np.pi, 30)
    prices_import = np.sin(x)

    batteries = [
        Battery(
            id=i,
            capacity=40,
            energy_start=10,
            energy_end=40,
            energy_min=5,
            energy_max=40,
            power_charge_max=10,
            power_discharge_max=10,
            connected=[True] * 30,
        )
        for i in range(1000)
    ]

    batteries_2 = [
        Battery(
            id=i,
            capacity=40,
            energy_start=12,
            energy_end=40,
            energy_min=5,
            energy_max=40,
            power_charge_max=10,
            power_discharge_max=10,
            connected=[False] * 5 + [True] * 25,
        )
        for i in range(1000)
    ]

    fo1 = FleetOptimizationOR(id=1)
    for battery in batteries:
        fo1.add_battery(battery)
    fo1.add_prices(prices_import, prices_import)

    fo2 = FleetOptimizationOR(id=2)
    for battery in batteries_2:
        fo2.add_battery(battery)
    fo2.add_prices(prices_import, prices_import)

    pool = PoolOptimizer(
        fleet_optimizer=[fo1, fo2], marketed_volumes=np.array([5] * 5 + [np.nan] * 25)
    )

    result = pool.optimize(type="simple")
    for id, site in result["sites"].items():
        print(site.head())

    print(result["pool"].head())
