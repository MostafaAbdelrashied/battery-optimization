from dataclasses import asdict, dataclass
from typing import Dict


@dataclass
class Grid:
    """
    Represents a power grid with feed and purchase capabilities.

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

    feed_power_limit: float
    purchase_power_limit: float
    purchase_efficiency: float = 1.0
    feed_efficiency: float = 1.0

    def __post_init__(self):
        if self.feed_power_limit < 0:
            raise ValueError("feed_power_limit must be non-negative")
        if self.purchase_power_limit < 0:
            raise ValueError("purchase_power_limit must be non-negative")
        if not 0 < self.purchase_efficiency <= 1:
            raise ValueError("purchase_efficiency must be between 0 and 1")
        if not 0 < self.feed_efficiency <= 1:
            raise ValueError("feed_efficiency must be between 0 and 1")

    def dict(self) -> Dict[str, float]:
        """
        Returns a dictionary representation of the Grid object.

        Returns
        -------
        Dict[str, float]
            A dictionary containing the Grid object's attributes.
        """
        return asdict(self)

    def __str__(self) -> str:
        return (
            f"Grid(feed_power_limit={self.feed_power_limit}, "
            f"purchase_power_limit={self.purchase_power_limit}, "
            f"feed_efficiency={self.feed_efficiency}, "
            f"purchase_efficiency={self.purchase_efficiency})"
        )


if __name__ == "__main__":
    grid = Grid(
        feed_power_limit=100,
        purchase_power_limit=200,
        purchase_efficiency=1,
        feed_efficiency=0.9,
    )
    print(grid.dict())
