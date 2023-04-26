import datetime

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import collections, figure


def plot_colormesh(
    X: np.ndarray,
    P: np.ndarray,
    Z: np.ndarray,
    plot_limit: float = 10,
    three_d_plot: bool = False,
    show_plot: bool = True,
    save_plot: bool = False,
    plot_name: str = "",
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
            ax.axes.set_xlim3d(left=-plot_limit, right=plot_limit)
            ax.axes.set_ylim3d(bottom=-plot_limit, top=plot_limit)
        else:
            ax.axis(
                [
                    -plot_limit,
                    plot_limit,
                    -plot_limit,
                    plot_limit,
                ]
            )

    fig = plt.figure()
    ax = get_axis(fig, three_d_plot)
    c = plot_collection(ax, X, P, Z)
    add_labels(ax)
    add_limits(ax)
    fig.colorbar(c, ax=ax)
    ax.set_aspect(aspect=1, adjustable="box")

    # fig.set_size_inches(4.8, 5)
    # ax.set_axis_off()

    if save_plot:
        fig.savefig(plot_name or f"{datetime.datetime.utcnow()}.png")

    if show_plot:
        plt.show()

    plt.close(fig=fig)
