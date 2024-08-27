from abc import abstractmethod

import pandas as pd
from loguru import logger

from battery_management.assets.asset_status import AssetStatus
from battery_management.assets.charging_point import ChargingPoint
from battery_management.assets.site import Site
from battery_management.request_handler.pool_optimizer import PoolOptimizer
from battery_management.response_handler.response_handler import ResponseHandler


class RequestHandler:
    """
    This is based on experiences with VPP requests. The main functionality is to receive a VPP request in JSON
    format, calculate an optimised charging schedule (steering or day-ahead) and save a response in JSON format
    """

    def __init__(self, request: dict):
        self.request = request

        # Parsing the header is standard
        self.start_time = pd.Timestamp(
            pd.Timestamp(request["request"]["start_time"]).value
        )
        self.end_time = pd.Timestamp(pd.Timestamp(request["request"]["end_time"]).value)
        self.id = request["request"]["id"]

        # Optimisation Result
        self.result = None

        # For saving files
        self.root_dir = None

        # Optional date-time index
        self.optimization_period = None

        # Set ResponseHandler and PoolOptimiser class. This can be overwritten in individual implementations
        self.response_handler = ResponseHandler
        self.pool_optimizer = PoolOptimizer

        self.parse(request)

    @abstractmethod
    def parse(self, request: dict):
        """
        This is the method to parse the request. At this point it is too project-specific to unify it. It will be
        executed at the end of the constructor so it must be overwritten in each project. An example would be to
        have an implementation like

        self.sites = [Site(s) for s in request['sites']]

        This way we can also define Site (and within assets) on project level. Ideally, mid-term this can be replaced
        with a more unified version here that is only overwritten where necessary.

        Returns
        -------

        """
        pass

    def optimize(self) -> ResponseHandler:
        """
        Create Pool, execute optimization, receive results
        This is a minimal implementation. In all likelihood individual projects need to overwrite parts here and
        in the private methods used

        Returns
        -------

        """
        logger.debug("-" * 125)
        logger.debug("Setup Site Optimizer")
        logger.debug("-" * 125)

        # Disconnected Sites ---------------

        disconnected_site_optimizer = [
            y
            for y in (self._create_site_optimizer(site) for site in self.sites)
            if y is not None
        ]
        if len(disconnected_site_optimizer) > 0:
            disconnected = self.pool_optimizer(
                disconnected_site_optimizer, datetime_index=self.optimization_period
            )
            result_disconnected = disconnected.optimize()
        else:
            result_disconnected = None

        return self.response_handler(request=self.request, results=result_disconnected)

    def _create_site_optimizer(self, site: Site):
        raise NotImplementedError()

    def _create_battery(
        self, i: int, site: Site, charging_point: ChargingPoint, status: AssetStatus
    ):
        raise NotImplementedError()

    def _get_prices(self, site: Site):
        raise NotImplementedError()


if __name__ == "__main__":
    # Let's test this thing!
    test_request = {
        "id": 42,
        "start_time": pd.Timestamp(2020, 8, 31, 10),
        "end_time": pd.Timestamp(2020, 8, 31, 10),
    }
    test_request = {"request": test_request}

    req1 = RequestHandler(test_request)

    stationary_battery = dict(
        id=42, energy_min=5, energy_max=40, power_charge_max=5, power_discharge_max=5
    )
    site1 = {"site_id": 12, "stationary_batteries": [stationary_battery]}
    test_request["site_specifications"] = [site1]

    req2 = RequestHandler(test_request)

    print("foo")
