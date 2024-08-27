import json
import os
from abc import ABC, abstractmethod
from typing import Any, Dict

import pandas as pd
from loguru import logger

from battery_management.helper.pool_result import PoolResult

RESULT_DIR = "results"


class ResponseHandler(ABC):
    """
    Wrapper for the PoolResult class to generate a controlled JSON response with a clear format.
    """

    def __init__(
        self,
        request: Dict[str, Any],
        results: PoolResult,
        request_type: str = "baseclass",
    ):
        """
        Initialize the ResponseHandler.

        Parameters
        ----------
        request : Dict[str, Any]
            The original request dictionary.
        results : PoolResult
            The results of the pool optimization.
        request_type : str, optional
            The type of request, by default "baseclass".
        """
        self.request_type = request_type
        self.root_dir: str = ""
        self.request = request
        self.start_time = pd.Timestamp(request["request"]["start_time"])
        self.end_time = pd.Timestamp(request["request"]["end_time"])
        self.id = request["request"]["id"]
        self.pool_results = results
        now = pd.Timestamp.utcnow()

        self.filepath = f'data/results/{now.strftime("%Y/%m/%d")}/{self.id}'
        if self.root_dir:
            self.filepath = f"{self.root_dir}/{RESULT_DIR}"
        self.filename = (
            f'{self.request_type}_response_{now.strftime("%Y%m%d%H%M%S")}.json'
        )
        self.response = self.create_response(results)

    def create_response(self, results: PoolResult) -> Dict[str, Any]:
        """
        Create a response dictionary from the results.

        Parameters
        ----------
        results : PoolResult
            The results of the pool optimization.

        Returns
        -------
        Dict[str, Any]
            The created response dictionary.
        """
        if results is None:
            return {}

        response = {
            "request": {
                "start_time": f"{self.start_time.isoformat()}Z",
                "end_time": f"{self.end_time.isoformat()}Z",
                "id": self.id,
            }
        }

        response = self.save_result(response, results)
        self.save_to_json(response)

        return response

    def save_to_json(self, response: Dict[str, Any]) -> None:
        """
        Save the response dictionary to a JSON file.

        Parameters
        ----------
        response : Dict[str, Any]
            The response dictionary to save.
        """
        if not os.path.exists(self.filepath):
            os.makedirs(self.filepath)

        with open(f"{self.filepath}/{self.filename}", "w") as f:
            json.dump(response, f)
        logger.debug(f"Saved response as {self.filename}")

    @abstractmethod
    def save_result(
        self, response: Dict[str, Any], results: PoolResult
    ) -> Dict[str, Any]:
        """
                Save the result to the response dictionary. Must be implemented in subclasses.

                Parameters
                ----------
                response : Dict[str, Any]
                    The response dictionary to update.
                results : PoolResult
                    The results of the pool optimization.
        Returns
                -------
                Dict[str, Any]
                    The updated response dictionary.
        """
        pass

    @staticmethod
    def _row_to_json(
        row: pd.Series, columns: Dict[str, Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        Convert a DataFrame row to a JSON-like dictionary.

        Parameters
        ----------
        row : pd.Series
            The DataFrame row to convert.
        columns : Dict[str, Dict[str, Any]]
            A dictionary specifying how to convert each column.

        Returns
        -------
        Dict[str, Any]
            The converted row as a dictionary.
        """
        result = {
            "start_time": f'{row["start_time"].isoformat()}Z',
            "end_time": f'{row["end_time"].isoformat()}Z',
        }
        for k, sub_dict in columns.items():
            value = row[k]
            result[sub_dict["output_name"]] = value * sub_dict.get("factor", 1)

        return result


class V2GSteeringResponse(ResponseHandler):
    """
    Response handler for V2G Steering requests.
    """

    request_type = "V2GSteering"

    def save_result(
        self, response: Dict[str, Any], results: PoolResult
    ) -> Dict[str, Any]:
        """
        Save the V2G Steering result to the response dictionary.

        Parameters
        ----------
        response : Dict[str, Any]
            The response dictionary to update.
        results : PoolResult
            The results of the pool optimization.

        Returns
        -------
        Dict[str, Any]
            The updated response dictionary.
        """
        assets = []
        for i, ddf in results.battery_results.items():
            if ddf.empty:
                continue
            df = ddf.copy()
            site_id = int(df["site_id"].unique()[0])
            df = df[df["power_kw"].notna()]

            df["start_time"] = df.index
            df["end_time"] = df["start_time"] + pd.Timedelta("30min")
            df = df[
                (df["start_time"] >= self.start_time)
                & (df["end_time"] <= self.end_time)
            ]
            if df.empty:
                continue

            columns = {"power_kw": {"output_name": "basepoint_kw", "factor": -1}}
            df["json"] = df.apply(lambda row: self._row_to_json(row, columns), axis=1)
            asset = {"site_id": site_id, "asset_id": i, "data": list(df["json"])}
            assets.append(asset)

        response["assets"] = assets
        return response


class V2GOptimisationResponse(ResponseHandler):
    """
    Response handler for V2G Optimisation requests.
    """

    request_type = "V2GOptimisation"

    def save_result(
        self, response: Dict[str, Any], results: PoolResult
    ) -> Dict[str, Any]:
        """
        Save the V2G Optimisation result to the response dictionary.

        Parameters
        ----------
        response : Dict[str, Any]
            The response dictionary to update.
        results : PoolResult
            The results of the pool optimization.

        Returns
        -------
        Dict[str, Any]
            The updated response dictionary.
        """
        df = results.pool_results.copy()
        df = df.loc[self.start_time : self.end_time]

        df["start_time"] = df.index
        df["end_time"] = df["start_time"] + pd.Timedelta("30min")
        df = df[
            (df["start_time"] >= self.start_time) & (df["end_time"] <= self.end_time)
        ]

        columns = {"power_kw": {"output_name": "volume_kw", "factor": -1}}
        df["json"] = df.apply(lambda row: self._row_to_json(row, columns), axis=1)
        response["data"] = [{"market": "DayAhead", "data": list(df["json"])}]
        return response


if __name__ == "__main__":
    # Example usage
    example_request = {
        "request": {
            "start_time": "2023-05-01T00:00:00Z",
            "end_time": "2023-05-01T23:59:59Z",
            "id": "example_id",
        }
    }
    example_results = PoolResult({})  # Empty PoolResult for demonstration

    v2g_steering = V2GSteeringResponse(example_request, example_results)
    v2g_optimisation = V2GOptimisationResponse(example_request, example_results)

    print("V2G Steering Response:", v2g_steering.response)
    print("V2G Optimisation Response:", v2g_optimisation.response)
