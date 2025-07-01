import typing
from typing import Optional, Union, Literal
from dataclasses import dataclass
import warnings
import abc

import numpy as np
import scipy.optimize as opt

import sets




class BasicLogger:
    def __init__(self) -> None:
        self.data = {}

    def log(self, key, value):
        if key not in self.data:
            self.data[key] = np.array([value])
        else:
            self.data[key] = np.append(self.data[key], [value], axis=0)

    def __getitem__(self, key):
        return self.data[key]


class FunctionWithGradient:
    def __init__(self, f, grad_f):
        super().__init__()

        self.f = f
        self.grad_f = grad_f

    def __call__(self, arg):
        return self.f(arg)

    def grad(self, arg):
        return self.grad_f(arg)


class GradientStopper(abc.ABC):
    @abc.abstractmethod
    def __call__(self, old_pos, new_pos, old_val, new_val): ...


class PointStopper(GradientStopper):
    def __init__(self, eps):
        super().__init__()

        self.eps = eps

    def __call__(self, old_pos, new_pos, old_val, new_val):
        return np.linalg.norm(old_pos - new_pos) < self.eps


class UnitSphere:
    def projection(self, arg):
        return np.array(arg) / np.linalg.norm(arg)


@dataclass
class StepConst:
    alpha: float


@dataclass
class StepFastest:
    pass


@dataclass
class StepStandard:
    gamma: float


ProjGradStep = StepConst
CondGradStep = StepConst | StepFastest | StepStandard


def gradient_projection(
    f: FunctionWithGradient,
    Q: Union[sets.ConvexSet, UnitSphere],
    init_point,
    stopping: GradientStopper = PointStopper(1e-6),
    step: ProjGradStep = StepConst(1e-3),
    max_iter=5000,
    logger: Optional[BasicLogger] = None,
):
    x = np.array(init_point)
    value = f(x)

    if logger:
        logger.log("x", x)
        logger.log("val", value)
        logger.log("grad", f.grad(x))

    for _ in range(max_iter):
        grad = f.grad(x)
        x_next = Q.projection(x - step.alpha * grad)
        value_next = f(x_next)

        if logger:
            logger.log("x", x_next)
            logger.log("val", value_next)
            logger.log("grad", f.grad(x_next))

        if stopping(x, x_next, value, value_next):
            return x_next

        x = x_next
        value = value_next

    warnings.warn(f"gradient_projection reached max_iter = {max_iter}")
    return x


def conditional_gradient(
    f: FunctionWithGradient,
    Q: sets.ConvexSet,
    init_point,
    stopping: GradientStopper = PointStopper(1e-6),
    step: CondGradStep = StepConst(1e-3),
    max_iter=5000,
    logger: Optional[BasicLogger] = None,
):
    x = np.array(init_point)
    value = f(x)

    if logger:
        logger.log("x", x)
        logger.log("val", value)

    for _ in range(max_iter):
        linear_solution = Q(-f.grad(x))

        polyak_coef = (
            np.dot(f.grad(x), x - linear_solution)
            / np.linalg.norm(x - linear_solution) ** 2
        )

        if logger:
            logger.log("lin_sol", linear_solution)
            logger.log("pol_coef", polyak_coef)

        match step:
            case StepConst(step_alpha):
                alpha = step_alpha
            case StepFastest():
                f_on_line = lambda t: f((1 - t) * x + t * linear_solution)
                opt_result = opt.minimize_scalar(
                    f_on_line, bounds=(0, 1), method="bounded"
                )

                if not opt_result.success:
                    warnings.warn(
                        f"linear optimization failed inside conditional_gradient"
                    )
                alpha = opt_result.x
            case StepStandard(gamma):
                alpha = min(1, gamma * polyak_coef)
            case _:
                typing.assert_never(step)

        next_x = alpha * linear_solution + (1 - alpha) * x
        next_value = f(next_x)

        if logger:
            logger.log("alpha", alpha)
            # next iteration
            logger.log("x", next_x)
            logger.log("val", next_value)

        if stopping(x, next_x, value, next_value):
            return next_x

        x = next_x
        value = next_value

    warnings.warn(f"conditional_gradient reached max_iter = {max_iter}")
    return x
