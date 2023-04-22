from msc_misc.kerr_interaction.circuits import kerr_displace_squeeze_bressanini
from msc_misc.plot_utils.plot_colormesh import plot_colormesh

if __name__ == "__main__":
    quadrature_plot_limit = 13
    X, P, Z = kerr_displace_squeeze_bressanini(
        squeezing=1,
        squeezing_phase=0,
        displacement=5,
        kerr_integer=5,
        fock_cutoff_dim=30,
        x_p_quadratures_limit=quadrature_plot_limit,
        x_p_number_of_points=200,
    )
    plot_colormesh(
        X,
        P,
        Z,
        plot_limit=quadrature_plot_limit,
        three_d_plot=False,
    )
