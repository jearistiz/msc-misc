from msc_misc.kerr_interaction.circuits import kerr_displace_squeeze_bressanini
from msc_misc.plot_utils.plot_colormesh import plot_colormesh
from msc_misc.simulation_utils import wigner_samples_from_simulation

PHASE_SPACE_PLOT_LIMIT = 13

if __name__ == "__main__":
    simulation = kerr_displace_squeeze_bressanini(
        squeezing=1,
        squeezing_phase=0,
        displacement=5,
        kerr_integer=5,
        fock_cutoff_dim=30,
    )
    X, P, Z = wigner_samples_from_simulation(
        simulation=simulation,
        x_p_limit=PHASE_SPACE_PLOT_LIMIT,
        x_p_number_of_points=200,
    )
    plot_colormesh(X, P, Z, plot_limit=PHASE_SPACE_PLOT_LIMIT, three_d_plot=False)
