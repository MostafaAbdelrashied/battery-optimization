from typing import Dict


class Grid:
    """
    Represents a power grid with feed and purchase capabilities.

    Parameters
    ----------
    feed_power_limit : float
        The limit of the power that can be fed into the grid in W.
    purchase_power_limit : float
        The limit of the power that can be purchased from the grid in W.
    purchase_efficiency : float, optional
        The efficiency of purchasing power from the grid (default is 1).
    feed_efficiency : float, optional
        The efficiency of feeding power into the grid (default is 1).

    Attributes
    ----------
    feed_power_limit : float
        The limit of the power that can be fed into the grid in W.
    purchase_power_limit : float
        The limit of the power that can be purchased from the grid in W.
    purchase_efficiency : float
        The efficiency of purchasing power from the grid.
    feed_efficiency : float
        The efficiency of feeding power into the grid.
    """

    def __init__(
        self,
        feed_power_limit: float,
        purchase_power_limit: float,
        purchase_efficiency: float = 1,
        feed_efficiency: float = 1,
    ):
        self.feed_power_limit = feed_power_limit
        self.purchase_power_limit = purchase_power_limit
        self.purchase_efficiency = purchase_efficiency
        self.feed_efficiency = feed_efficiency

    def dict(self) -> Dict[str, float]:
        """
        Returns a dictionary representation of the Grid object.

        Returns
        -------
        Dict[str, float]
            A dictionary containing the Grid object's attributes.
        """
        return self.__dict__


if __name__ == "__main__":
    grid = Grid(
        feed_power_limit=100,
        purchase_power_limit=200,
        purchase_efficiency=1,
        feed_efficiency=0.9,
    )
    print(grid.dict())
