"""Plots for states in Bressanini, 2022 PhysRevA.106.042413"""
import matplotlib.pyplot as plt
import numpy as np
import strawberryfields as sf
from matplotlib import collections, figure
from strawberryfields import ops


def run_circuit():
    squeeze_displace_kerr(
        squeezing=1,
        displacement=2,
        kerr_integer=2,
        fock_cutoff_dim=30,
        quadrature_plot_limit=10,
        grid_plot_linear_points=150,
        three_d_plot=False,
    )


def squeeze_displace_kerr(
    squeezing: float = 0,
    displacement: float = 2,
    kerr_integer: int = 2,
    fock_cutoff_dim: int = 20,
    quadrature_plot_limit: float = 10,
    grid_plot_linear_points: int = 100,
    three_d_plot: bool = False,
) -> None:
    prog = sf.Program(1)

    with prog.context as q:
        ops.Squeezed(squeezing, np.pi / 2) | q[0]
        ops.Coherent(displacement, 0) | q[0]
        if kerr_integer != 0:
            ops.Kgate(np.pi / kerr_integer) | q[0]

    eng = sf.Engine("fock", backend_options={"cutoff_dim": fock_cutoff_dim})
    state = eng.run(prog).state

    X = np.linspace(
        -quadrature_plot_limit,
        quadrature_plot_limit,
        grid_plot_linear_points,
    )
    P = np.linspace(
        -quadrature_plot_limit,
        quadrature_plot_limit,
        grid_plot_linear_points,
    )
    Z = state.wigner(0, X, P)
    X, P = np.meshgrid(X, P)

    def plot_figure():
        fig = plt.figure()
        ax = get_axis(fig)
        c = plot_collection(ax)
        add_labels(ax)
        add_limits(ax)
        fig.colorbar(c, ax=ax)
        # fig.set_size_inches(4.8, 5)
        # ax.set_axis_off()
        plt.show()

    def get_axis(fig: figure.Figure) -> plt.Axes:
        if three_d_plot:
            from mpl_toolkits.mplot3d import Axes3D

            return fig.add_subplot(111, projection="3d")
        else:
            return fig.add_subplot(111)

    def plot_collection(ax: plt.Axes) -> collections.Collection:
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

    plot_figure()


if __name__ == "__main__":
    run_circuit()
