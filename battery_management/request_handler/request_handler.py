from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional

import pandas as pd
from loguru import logger

from battery_management.assets.asset_status import AssetStatus
from battery_management.assets.charging_point import ChargingPoint
from battery_management.assets.site import Site
from battery_management.request_handler.pool_optimizer import PoolOptimizer
from battery_management.response_handler.response_handler import ResponseHandler


class RequestHandler(ABC):
    """
    Base class for handling VPP requests, calculating optimized charging schedules,
    and saving responses in JSON format.
    """

    def __init__(self, request: Dict[str, Any]):
        """
        Initialize the RequestHandler.

        Parameters
        ----------
        request : Dict[str, Any]
            The VPP request in JSON format.
        """
        self.request = request
        self.start_time = pd.Timestamp(request["request"]["start_time"])
        self.end_time = pd.Timestamp(request["request"]["end_time"])
        self.id = request["request"]["id"]
        self.result: Optional[Any] = None
        self.root_dir: Optional[str] = None
        self.optimization_period: Optional[pd.DatetimeIndex] = None
        self.response_handler = ResponseHandler
        self.pool_optimizer = PoolOptimizer
        self.sites: List[Site] = []

        self.parse(request)

    @abstractmethod
    def parse(self, request: Dict[str, Any]) -> None:
        """
        Parse the request. Must be implemented in subclasses.

        Parameters
        ----------
        request : Dict[str, Any]
            The VPP request in JSON format.
        """
        pass

    def optimize(self) -> ResponseHandler:
        """
        Create Pool, execute optimization, and receive results.

        Returns
        -------
        ResponseHandler
            The response handler containing optimization results.
        """
        logger.debug("-" * 125)
        logger.debug("Setup Site Optimizer")
        logger.debug("-" * 125)

        disconnected_site_optimizer = [
            y
            for y in (self._create_site_optimizer(site) for site in self.sites)
            if y is not None
        ]
        if disconnected_site_optimizer:
            disconnected = self.pool_optimizer(
                disconnected_site_optimizer, datetime_index=self.optimization_period
            )
            result_disconnected = disconnected.optimize()
        else:
            result_disconnected = None

        return self.response_handler(request=self.request, results=result_disconnected)

    @abstractmethod
    def _create_site_optimizer(self, site: Site) -> Any:
        """
        Create a site optimizer. Must be implemented in subclasses.

        Parameters
        ----------
        site : Site
            The site to create an optimizer for.

        Returns
        -------
        Any
            The created site optimizer.
        """
        raise NotImplementedError()

    @abstractmethod
    def _create_battery(
        self, i: int, site: Site, charging_point: ChargingPoint, status: AssetStatus
    ) -> Any:
        """
        Create a battery. Must be implemented in subclasses.

        Parameters
        ----------
        i : int
            Battery identifier.
        site : Site
            The site containing the battery.
        charging_point : ChargingPoint
            The charging point for the battery.
        status : AssetStatus
            The status of the asset.

        Returns
        -------
        Any
            The created battery.
        """
        raise NotImplementedError()

    @abstractmethod
    def _get_prices(self, site: Site) -> Any:
        """
        Get prices for a site. Must be implemented in subclasses.

        Parameters
        ----------
        site : Site
            The site to get prices for.

        Returns
        -------
        Any
            The prices for the site.
        """
        raise NotImplementedError()


if __name__ == "__main__":
    test_request = {
        "request": {
            "id": 42,
            "start_time": pd.Timestamp(2020, 8, 31, 10),
            "end_time": pd.Timestamp(2020, 8, 31, 10),
        },
        "site_specifications": [
            {
                "site_id": 12,
                "stationary_batteries": [
                    {
                        "id": 42,
                        "energy_min": 5,
                        "energy_max": 40,
                        "power_charge_max": 5,
                        "power_discharge_max": 5,
                    }
                ],
            }
        ],
    }

    class TestRequestHandler(RequestHandler):
        def parse(self, request: Dict[str, Any]) -> None:
            self.sites = [Site(s) for s in request["site_specifications"]]

        def _create_site_optimizer(self, site: Site) -> None:
            pass

        def _create_battery(
            self, i: int, site: Site, charging_point: ChargingPoint, status: AssetStatus
        ) -> None:
            pass

        def _get_prices(self, site: Site) -> None:
            pass

    req = TestRequestHandler(test_request)
    print("RequestHandler initialized successfully")
