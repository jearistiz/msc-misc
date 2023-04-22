"""Plots for states in Bressanini, 2022 PhysRevA.106.042413"""
import matplotlib.pyplot as plt
import numpy as np
import strawberryfields as sf
from matplotlib import collections, figure
from strawberryfields import ops


def run_circuit():
    quadrature_plot_limit = 13
    X, P, Z = squeeze_displace_kerr(
        squeezing=1,
        squeezing_phase=0,
        displacement=5,
        kerr_integer=5,
        fock_cutoff_dim=30,
        x_p_quadratures_limit=quadrature_plot_limit,
        x_p_number_of_points=200,
    )
    plot_figure(
        X,
        P,
        Z,
        quadrature_plot_limit=quadrature_plot_limit,
        three_d_plot=False,
    )


def squeeze_displace_kerr(
    squeezing: float = 0,
    squeezing_phase: float = np.pi / 2,
    displacement: float = 2,
    kerr_integer: int = 2,
    fock_cutoff_dim: int = 20,
    x_p_quadratures_limit: float = 10,
    x_p_number_of_points: int = 200,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
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


def plot_figure(
    X: np.ndarray,
    P: np.ndarray,
    Z: np.ndarray,
    quadrature_plot_limit: float = 10,
    three_d_plot: bool = False,
) -> None:
    def get_axis(fig: figure.Figure, three_d_plot: bool = False) -> plt.Axes:
        if three_d_plot:
            from mpl_toolkits.mplot3d import Axes3D

            return fig.add_subplot(111, projection="3d")
        else:
            return fig.add_subplot(111)

    def plot_collection(
        ax: plt.Axes, X: np.ndarray, P: np.ndarray, Z: np.ndarray
    ) -> collections.Collection:
        if three_d_plot:
            return ax.plot_surface(X, P, Z, cmap="RdYlGn", lw=0.5, rstride=1, cstride=1)
        else:
            return ax.pcolormesh(X, P, Z, cmap="RdYlGn")

    def add_labels(ax: plt.Axes):
        ax.set_xlabel("X")
        ax.set_ylabel("P")
        if three_d_plot:
            ax.set_zlabel("W")

    def add_limits(ax: plt.Axes):
        if three_d_plot:
            ax.axes.set_xlim3d(left=-quadrature_plot_limit, right=quadrature_plot_limit)
            ax.axes.set_ylim3d(bottom=-quadrature_plot_limit, top=quadrature_plot_limit)
        else:
            ax.axis(
                [
                    -quadrature_plot_limit,
                    quadrature_plot_limit,
                    -quadrature_plot_limit,
                    quadrature_plot_limit,
                ]
            )

    fig = plt.figure()
    ax = get_axis(fig, three_d_plot)
    c = plot_collection(ax, X, P, Z)
    add_labels(ax)
    add_limits(ax)
    fig.colorbar(c, ax=ax)
    # fig.set_size_inches(4.8, 5)
    # ax.set_axis_off()
    plt.show()


if __name__ == "__main__":
    run_circuit()
