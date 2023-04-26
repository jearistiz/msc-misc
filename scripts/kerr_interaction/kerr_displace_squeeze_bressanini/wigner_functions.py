import numpy as np

from msc_misc.kerr_interaction.circuits import kerr_displace_squeeze_bressanini
from msc_misc.plot_utils.plot_colormesh import plot_colormesh
from msc_misc.simulation_utils import wigner_samples_from_simulation

TOLERANCE = 0.0005
ALPHAS = [0, 1, 3, 5]
SQUEEZINGS = [0, 0.2, 0.4, 0.6]  # TODO: Ask about these values and the dB units
X_P_NUMBER_OF_POINTS = 200
START_CUTOFF_DIMENSION = 20
MAX_CUTOFF_DIMENSION = 54
CUTOFF_DIMENSION_DELTA = 2
PHASE_SPACE_PLOT_LIMIT = 14
"""
NOTES
----
* If we use more than r=1.2, the cutoff limit would need to be much higher.
* After cutoff = 70 the noise is very high and apparently the simulation
  breaks i.e. we get nonsense plots
* Found that 15 dB was the maximum measured ~ r = 1.73 (4 yrs ago)
https://physics.stackexchange.com/questions/411786/how-much-can-we-currently-squeeze-light
"""


def calculate_error(previous_z: np.ndarray, current_z: np.ndarray) -> np.ndarray:
    return np.abs(previous_z - current_z)


def find_convergent_state(
    squeezing: float,
    displacement: float,
    kerr_integer=2,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    cutoff_dim = START_CUTOFF_DIMENSION
    simulation = kerr_displace_squeeze_bressanini(
        squeezing=squeezing,
        squeezing_phase=0,
        displacement=displacement,
        kerr_integer=kerr_integer,
        fock_cutoff_dim=START_CUTOFF_DIMENSION,
    )
    previous_z = np.zeros((X_P_NUMBER_OF_POINTS, X_P_NUMBER_OF_POINTS))
    x, p, z = wigner_samples_from_simulation(
        simulation=simulation,
        x_p_limit=PHASE_SPACE_PLOT_LIMIT,
        x_p_number_of_points=200,
    )
    error = calculate_error(z, previous_z)

    while (not np.all(error < TOLERANCE)) and (cutoff_dim <= MAX_CUTOFF_DIMENSION):
        print(f"CUTOFF_DIM={cutoff_dim} AVG_ERR={np.average(error):.8f}")

        cutoff_dim += CUTOFF_DIMENSION_DELTA
        z = previous_z

        simulation = kerr_displace_squeeze_bressanini(
            squeezing=squeezing,
            squeezing_phase=0,
            displacement=displacement,
            kerr_integer=kerr_integer,
            fock_cutoff_dim=cutoff_dim,
        )
        x, p, z = wigner_samples_from_simulation(
            simulation=simulation,
            x_p_limit=PHASE_SPACE_PLOT_LIMIT,
            x_p_number_of_points=200,
        )

        error = calculate_error(z, previous_z)

    plot_colormesh(x, p, z, plot_limit=PHASE_SPACE_PLOT_LIMIT, three_d_plot=False)


def plot_all(
    displacements: list[float],
    squeezings: list[float],
    squeezing_phase_pi_units: float = 0,
    kerr_integer: int = 2,
    fock_cutoff_dim: int = 70,
    x_p_limit: int = PHASE_SPACE_PLOT_LIMIT,
    x_p_number_of_points: int = 400,
) -> None:
    for alpha in displacements:
        for squeezing in squeezings:
            simulation = kerr_displace_squeeze_bressanini(
                squeezing=squeezing,
                squeezing_phase=squeezing_phase_pi_units * np.pi,
                displacement=alpha,
                kerr_integer=kerr_integer,
                fock_cutoff_dim=fock_cutoff_dim,
            )
            x, p, z = wigner_samples_from_simulation(
                simulation=simulation,
                x_p_limit=x_p_limit,
                x_p_number_of_points=x_p_number_of_points,
            )
            plot_colormesh(
                x,
                p,
                z,
                plot_limit=x_p_limit,
                show_plot=False,
                save_plot=True,
                plot_name="".join(
                    [
                        f"kerr_integer_{kerr_integer}-",
                        f"squeezing_phase_pi_units_{squeezing_phase_pi_units:.4f}-",
                        f"displacement_{alpha:.4f}-",
                        f"squeezing_{squeezing:.4f}-"
                        f"fock_cutoff_dim_{fock_cutoff_dim}-",
                        f"x_p_limit_{x_p_limit}-",
                        f"x_p_number_of_points_{x_p_number_of_points}",
                        ".png",
                    ]
                ),
            )


if __name__ == "__main__":
    # find_convergent_state(1, 3)
    for kerr_integer in range(2, 6):
        plot_all(
            displacements=ALPHAS,
            squeezings=SQUEEZINGS,
            kerr_integer=kerr_integer,
            squeezing_phase_pi_units=1,
        )
