import numpy as np
import sortedcontainers as sc
import abc


def spherical_to_cartesian(r, phi, theta):
    return (
        np.array(
            [np.cos(phi) * np.cos(theta), np.sin(phi) * np.cos(theta), np.sin(theta)]
        )
        * r
    )


def fibonacci_lattice(n):
    golden_ratio = (1 + np.sqrt(5)) / 2
    idx = np.arange(0, n)

    phi = 2 * np.pi * idx / golden_ratio
    theta = np.arccos(1 - 2 * (idx + 0.5) / n)

    xs, ys, zs = np.cos(phi) * np.sin(theta), np.sin(phi) * np.sin(theta), np.cos(theta)

    return np.vstack([xs, ys, zs]).transpose()


def spherical_grid(phi_lim, theta_lim, n):
    phis = np.linspace(*phi_lim, n) * np.pi / 180
    thetas = np.linspace(*theta_lim, n) * np.pi / 180

    p_grid, t_grid = np.meshgrid(phis, thetas)

    xs, ys, zs = (
        np.cos(p_grid) * np.sin(t_grid),
        np.sin(p_grid) * np.sin(t_grid),
        np.cos(t_grid),
    )

    return np.vstack([xs.flatten(), ys.flatten(), zs.flatten()]).transpose()


class Square:
    def __init__(self, corners):
        self.corners = corners

    def split_quad(self):
        new_points = [[self.corners[i], self.corners[i - 1]] for i in range(4)]
        new_points.append(list(self.corners))

        new_squares = [[0, 5, 8, 4], [5, 1, 6, 8], [8, 6, 2, 7], [4, 8, 7, 3]]

        return new_squares, new_points

    def split_binary(self, shift):
        corners_ord = list(np.roll(self.corners, -shift))
        idx_ord = list(np.roll(np.arange(4), -shift))
        new_points = [
            [corners_ord[0], corners_ord[3]],
            [corners_ord[1], corners_ord[2]],
        ]

        new_squares = [[idx_ord[0], idx_ord[1], 5, 4], [4, 5, idx_ord[2], idx_ord[3]]]

        return new_squares, new_points


class BaseLattice(abc.ABC):
    def __init__(self, conv_set, phi_lim=(-179, 180), theta_lim=(-90, 90)):
        self.set = conv_set

        self.squares = sc.SortedList(key=self.get_size)
        self.lattice_pts = []
        self.support_pts = []

        p0, p1, t0, t1 = np.array(phi_lim + theta_lim) * np.pi / 180

        points = [[p0, t0], [p1, t0], [p1, t1], [p0, t1]]
        idx = [self.create_point(phi, theta) for phi, theta in points]

        self.squares.add(Square(idx))

    def get_size(self, square):
        corners = square.corners
        supp_pts = self.support_pts

        res = 0
        for i in range(4):
            res = max(
                res, np.linalg.norm(supp_pts[corners[i]] - supp_pts[corners[i - 1]])
            )

        return res

    def create_point(self, phi, theta):
        self.lattice_pts.append(np.array([phi, theta]))

        p = spherical_to_cartesian(1, phi, theta)
        self.support_pts.append(self.set(p))

        return len(self.support_pts) - 1

    @abc.abstractmethod
    def max_square(self):
        pass

    def get_new_squares(self, base_square, splits):
        new_squares, averages = splits

        point_ids = base_square.corners
        for idx in averages:
            new_point = np.mean([self.lattice_pts[i] for i in idx], axis=0)
            point_ids.append(self.create_point(*new_point))

        result = []
        for idx in new_squares:
            result.append(Square([point_ids[i] for i in idx]))
        return result

    @abc.abstractmethod
    def split_leaf(self, max_square):
        pass

    def split_all(self):
        old_squares = list(self.squares)
        new_squares = []

        for sq in old_squares:
            new_squares.extend(self.get_new_squares(sq, sq.split_quad()))

        self.squares.clear()
        self.squares.update(new_squares)

    def split_all_n(self, n):
        for i in range(n):
            self.split_all()

    def split_max(self):
        self.split_leaf(self.max_square())

    def get_unique(self, tol=1e-8):
        s_pts = np.array(self.support_pts)
        idx = np.lexsort((s_pts[:, 0], s_pts[:, 1], s_pts[:, 2]))

        idx_uniq = [idx[0]]
        for i in range(1, idx.shape[0]):
            if np.linalg.norm(s_pts[idx[i]] - s_pts[idx[i - 1]]) > tol:
                idx_uniq.append(idx[i])
        idx_uniq = np.array(idx_uniq)

        l_pts = np.array(self.lattice_pts)
        cart_latt = spherical_to_cartesian(1, l_pts[:, 0], l_pts[:, 1]).transpose()

        return l_pts[idx_uniq], cart_latt[idx_uniq], s_pts[idx_uniq]


class BinaryLattice(BaseLattice):
    def __init__(self, set, phi_lim=(-179, 180), theta_lim=(-90, 90)):
        super().__init__(set, phi_lim, theta_lim)

    def max_square(self):
        corners = self.squares[-1].corners
        supp_pts = self.support_pts

        lengths = []
        for i in range(4):
            lengths.append(
                np.linalg.norm(supp_pts[corners[i]] - supp_pts[corners[i - 1]])
            )

        return {"node_id": len(self.squares) - 1, "shift": np.argmax(lengths)}

    def split_leaf(self, max_square):
        base_square = self.squares[max_square["node_id"]]
        new_squares = self.get_new_squares(
            base_square, base_square.split_binary(max_square["shift"])
        )

        self.squares.pop(max_square["node_id"])
        for square in new_squares:
            self.squares.add(square)
