import numpy as np
import scipy.spatial as spat
import plotly.graph_objects as go
import matplotlib.pyplot as plt
from tqdm.rich import tqdm_rich
from tqdm import TqdmExperimentalWarning
import warnings
from lattice import BinaryLattice, fibonacci_lattice
from scipy.integrate import IntegrationWarning
import scipy.optimize as opt

warnings.filterwarnings("ignore", category=TqdmExperimentalWarning)
warnings.filterwarnings("ignore", category=IntegrationWarning)


def get_interior_point(X, lattice):
    norms = np.array([np.linalg.norm(p) for p in lattice])
    A = np.hstack([lattice, norms[:, np.newaxis]])
    b = [X.s(p) for p in lattice]

    c = np.zeros(len(lattice[0]) + 1)
    c[-1] = -1

    linprog_res = opt.linprog(c, A, b)
    if linprog_res.status == 0:
        return linprog_res.x[:-1]
    return None


def get_halfspaces(X, lattice):
    int_point = get_interior_point(X, lattice)
    if int_point is None:
        return None

    A = lattice
    b = np.array([-X.s(p) for p in lattice])

    halfspaces = np.hstack([A, b[:, np.newaxis]])
    return spat.HalfspaceIntersection(halfspaces, int_point)


def draw_set_2d(X, ax, color=None, N=200, strategy="elements", **kwargs):
    ps = np.array([[np.cos(t), np.sin(t)] for t in np.linspace(0, 2 * np.pi, N)])

    if strategy == "elements":
        boundary = np.array([X(p) for p in ps])
        ax.plot(boundary[:, 0], boundary[:, 1], c=color, **kwargs)
    elif strategy == "support":
        plot_2d_halfspaces(get_halfspaces(X, ps), ax, **kwargs)


def plot_2d_halfspaces(hs, ax, **kwargs):
    points = np.array(hs.intersections)
    sorted_points = sorted(points, key=lambda p: np.arctan2(*(p - hs.interior_point)))

    sorted_points.append(sorted_points[0])
    sorted_points = np.array(sorted_points)

    ax.plot(sorted_points[:, 0], sorted_points[:, 1], **kwargs)


def filter_hull(points, triangles, vertex_filter):
    pts_filter = np.array([vertex_filter(p) for p in points])
    tri_filter = np.array([np.all(pts_filter[s]) for s in triangles])

    repl = np.mean(points[pts_filter], axis=0)

    points[np.invert(pts_filter)] = repl
    triangles = triangles[tri_filter]

    return points, triangles


def draw_set_3d(
    X, fig=None, use_lattice=True, color=None, N=1000, vertex_filter=None, **kwargs
):
    if fig is None:
        fig = go.Figure()

    if not use_lattice:
        normals = fibonacci_lattice(N)
        points = np.array([X(p) for p in tqdm_rich(normals)])
    else:
        lattice = BinaryLattice(X)
        lattice.split_all_n(4)

        for i in tqdm_rich(range(N)):
            lattice.split_max()

        points = lattice.get_unique()[2]

    hull = spat.ConvexHull(points)
    points, triangles = hull.points, hull.simplices
    if vertex_filter is not None:
        points, triangles = filter_hull(points, triangles, vertex_filter)

    x, y, z = points.transpose()
    i, j, k = triangles.transpose()

    fig.add_mesh3d(
        x=x,
        y=y,
        z=z,
        i=i,
        j=j,
        k=k,
        opacity=0.5,
        hoverinfo="skip",
        legendrank=500,
        color=color,
        **kwargs
    )

    return fig
