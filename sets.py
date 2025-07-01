import abc

import numpy as np
import scipy.integrate as integ
import scipy.optimize as opt
import sympy as sp


class ConvexSet(abc.ABC):
    @abc.abstractmethod
    def __call__(self, p, *args): ...

    @abc.abstractmethod
    def s(self, p, *args): ...

    @property
    @abc.abstractmethod
    def dim(self): ...

    def projection(self, point, **kwargs):
        if "init_point" not in kwargs:
            kwargs["init_point"] = np.random.rand(self.dim)

        return projection_descent(self, point, **kwargs)

    def __add__(self, oth):
        return SumOfSets([self, oth])

    def multiply_by(self, mult):
        return MultSet(self, mult)


def projection_descent(Q: ConvexSet, point, **kwargs):
    import gradient

    f = gradient.LambdaFunction(
        lambda p: Q.s(p) - np.dot(p, point), lambda p: Q(p) - point
    )

    p0 = gradient.gradient_projection(f, gradient.UnitSphere(), **kwargs)
    if f(p0) <= 0:
        return point + p0 * f(p0)
    else:
        return point


class SetWithBoundArgs(ConvexSet):
    def __init__(self, conv_set: ConvexSet, set_args):
        super().__init__()

        self.conv_set = conv_set
        self.set_args = set_args

    @property
    def dim(self):
        return self.conv_set.dim

    def __call__(self, p, *args):
        return self.conv_set(p, *self.set_args, *args)

    def s(self, p, *args):
        return self.conv_set.s(p, *self.set_args, *args)


class Ellipse(ConvexSet):
    def __init__(self, A, x0):
        super().__init__()

        self.A_inv = np.linalg.inv(A)
        self.x0 = np.array(x0)

    @property
    def dim(self):
        return len(self.x0)

    def __call__(self, p, *_):
        p = np.array(p)
        return self.x0 + 1 / (np.sqrt(p @ self.A_inv @ p)) * self.A_inv @ p

    def s(self, p, *_):
        p = np.array(p)
        return np.dot(p, self.__call__(p))


class SumOfSets(ConvexSet):
    def __init__(self, sets):
        super().__init__()

        self.summands = sets

    @property
    def dim(self):
        return self.summands[0].dim

    def __call__(self, p, *args):
        return np.sum([S(p, *args) for S in self.summands], axis=0)

    def s(self, p, *args):
        return np.sum([S.s(p, *args) for S in self.summands])


class MultSet(ConvexSet):
    def __init__(self, conv_set: ConvexSet, multiplier):
        self.set = conv_set
        self.mult = multiplier

        if isinstance(multiplier, (int, float)):
            self.is_scalar = True
        elif hasattr(multiplier, "__len__"):
            self.is_scalar = False
            self.mult = np.array(self.mult)

    @property
    def dim(self):
        return self.set.dim

    def __call__(self, p):
        p = np.array(p)
        if self.is_scalar:
            return self.mult * self.set(self.mult * p)
        else:
            return self.mult @ self.set(np.transpose(self.mult) @ p)

    def s(self, p, *args):
        p = np.array(p)
        if self.is_scalar:
            return self.set.s(self.mult @ p, *args)
        else:
            return self.set.s(np.transpose(self.mult) @ p, *args)


class Point(ConvexSet):
    def __init__(self, point):
        super().__init__()

        self.point = point

    @property
    def dim(self):
        return len(self.point)

    def s(self, p, *args):
        return np.dot(p, self.point)

    def __call__(self, p, *args):
        return self.point


class Ball(ConvexSet):
    def __init__(self, center, radius):
        super().__init__()

        self.center = np.array(center)
        self.radius = radius

    @property
    def dim(self):
        return len(self.center)

    def s(self, p, *args):
        return np.linalg.norm(p) * self.radius + np.dot(p, self.center)

    def __call__(self, p, *args):
        p = np.array(p)
        if np.linalg.norm(p) == 0:
            return self.center
        return self.center + self.radius * p / np.linalg.norm(p)


class PNormBall(ConvexSet):
    def __init__(self, center, radius, p_norm):
        super().__init__()

        self.center = np.array(center)
        self.radius = np.array(radius)
        self.norm = p_norm

    @property
    def dim(self):
        return len(self.center)

    def s(self, p, *args):
        return np.dot(p, self.__call__(p, *args))

    def __call__(self, p, *args):
        p = np.array(p) / np.linalg.norm(p)
        direction = np.sign(p) * np.abs(p) ** (1 / (self.norm - 1))

        return self.center + self.radius * direction / np.linalg.norm(
            direction, ord=self.norm
        )


class ExampleAttSet(ConvexSet):
    def __init__(self):
        super().__init__()

    def __s_integrand_analytic(self, p, t):
        return (
            (1 / 2)
            * np.exp(-t)
            * (-(2 + 2 * t + t**2) * p[0] - 2 * (1 + t) * p[1] - 2 * p[2])
        )

    def __z_roots(self, p):
        a, b, c = p[0] / 2, p[1], p[2]
        if a == 0:
            if b == 0:
                return []
            else:
                return [-c / b]
        else:
            d = b**2 - 4 * a * c
            if d < 0:
                return []
            else:
                return [(-b + np.sqrt(d)) / (2 * a), (-b - np.sqrt(d)) / (2 * a)]

    def __continuity_segments(self, t, p):
        points = np.unique([0, t] + self.__z_roots(p))
        points = points[(points >= 0) & (points <= t)]

        q_z = lambda s: p[0] / 2 * s**2 + p[1] * s + p[2]
        signs = np.sign(q_z((points[:-1] + points[1:]) / 2))

        return points, signs

    @property
    def dim(self):
        return 3

    def s(self, p, t):
        points, signs = self.__continuity_segments(t, p)

        result = 0
        for i in range(signs.shape[0]):
            a, b = points[i], points[i + 1]

            result += signs[i] * (
                self.__s_integrand_analytic(p, b) - self.__s_integrand_analytic(p, a)
            )

        return result

    def __s_point_analytic(self, t):
        return np.array(
            [np.exp(-t) / 2 * (t**2 + 2 * t + 2), np.exp(-t) * (t + 1), np.exp(-t)]
        ) * (-1)

    def __call__(self, p, t):
        points, signs = self.__continuity_segments(t, p)

        result = np.zeros(3)
        for i in range(signs.shape[0]):
            a, b = points[i], points[i + 1]

            result += signs[i] * (
                self.__s_point_analytic(b) - self.__s_point_analytic(a)
            )

        return result
