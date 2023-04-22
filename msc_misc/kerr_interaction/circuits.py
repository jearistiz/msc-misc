import numpy as np
import strawberryfields as sf
from strawberryfields import ops


def kerr_displace_squeeze_bressanini(
    squeezing: float = 0,
    squeezing_phase: float = np.pi / 2,
    displacement: float = 2,
    kerr_integer: int = 2,
    fock_cutoff_dim: int = 20,
    x_p_quadratures_limit: float = 10,
    x_p_number_of_points: int = 200,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Calculates K(x)D(a)S(z)|0>
    Base Reference: Bressanini, 2022 PhysRevA.106.042413
    """
    prog = sf.Program(1)

    XI_BRESSANINI = np.pi / kerr_integer
    SQUEEZING_BRESSANINI = squeezing

    with prog.context as q:
        if squeezing:
            ops.Sgate(-SQUEEZING_BRESSANINI, squeezing_phase) | q[0]
        if displacement:
            ops.Dgate(displacement, 0) | q[0]
        if kerr_integer:
            ops.Kgate(-XI_BRESSANINI) | q[0]
            ops.Rgate(XI_BRESSANINI) | q[0]

    eng = sf.Engine("fock", backend_options={"cutoff_dim": fock_cutoff_dim})
    state = eng.run(prog).state

    X = np.linspace(
        -x_p_quadratures_limit,
        x_p_quadratures_limit,
        x_p_number_of_points,
    )
    P = np.linspace(
        -x_p_quadratures_limit,
        x_p_quadratures_limit,
        x_p_number_of_points,
    )
    Z = state.wigner(0, X, P)
    X, P = np.meshgrid(X, P)

    return X, P, Z
