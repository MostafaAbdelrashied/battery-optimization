from typing import Union

from battery_management.optimizer.battery_optimization_or import FleetOptimizationOR
from battery_management.optimizer.battery_optimization_scipy import (
    FleetOptimizationSciPy,
)


def create(
    id: int,
    type: str = "OR",
    calculate_savings: bool = False,
    dt: float = 1,
    fully_charged_as_penalty: bool = False,
    single_continuous_session_allowed: bool = False,
) -> Union[FleetOptimizationSciPy, FleetOptimizationOR]:
    """
    Factory method to create a battery optimizer.

    Parameters
    ----------
    id : int
        Identifier for the optimizer.
    type : str, optional
        Type of optimizer to create. Can be "SciPy" or "OR", by default "OR".
    calculate_savings : bool, optional
        Whether to calculate savings, by default False.
    dt : float, optional
        Time step for optimization, by default 1.
    fully_charged_as_penalty : bool, optional
        Whether to treat fully charged state as a penalty, by default False.
    single_continuous_session_allowed : bool, optional
        Whether to allow single continuous charging session, by default False.

    Returns
    -------
    Union[FleetOptimizationSciPy, FleetOptimizationOR]
        The created battery optimizer instance.

    Raises
    ------
    ValueError
        If an unsupported optimizer type is specified.
    """
    if type == "SciPy":
        return FleetOptimizationSciPy(id=id, calculate_savings=calculate_savings, dt=dt)
    elif type == "OR":
        return FleetOptimizationOR(
            id=id,
            calculate_savings=calculate_savings,
            dt=dt,
            fully_charged_as_penalty=fully_charged_as_penalty,
            single_continuous_session_allowed=single_continuous_session_allowed,
        )
    else:
        raise ValueError(f"Type {type} is not supported. Use 'SciPy' or 'OR'.")


if __name__ == "__main__":
    # Example usage
    optimizer_scipy = create(type="SciPy", id=1)
    print(f"Created SciPy optimizer: {optimizer_scipy}")

    optimizer_or = create(type="OR", id=2)
    print(f"Created OR optimizer: {optimizer_or}")

    try:
        create(type="MIP", id=3)
    except ValueError as e:
        print(f"Error creating unsupported optimizer: {e}")
