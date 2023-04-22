import numpy as np
from strawberryfields import result


def wigner_samples_from_simulation(
    simulation: result.Result,
    mode: int = 0,
    x_p_limit: float = 10,
    x_p_number_of_points: int = 200,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    X = np.linspace(
        -x_p_limit,
        x_p_limit,
        x_p_number_of_points,
    )
    P = np.linspace(
        -x_p_limit,
        x_p_limit,
        x_p_number_of_points,
    )

    assert (
        simulation.state is not None
    ), "Simulation must contain state in order to sample the Wigner function"

    Z = simulation.state.wigner(mode, X, P)
    X, P = np.meshgrid(X, P)

    return X, P, Z
