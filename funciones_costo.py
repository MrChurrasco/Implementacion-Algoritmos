import math
import numpy as np
from numpy.linalg import norm
from numba import vectorize, int64, float64

list_types_common = [int64(int64),
                     float64(float64)]

# Función Objetivo
def function_objetive(x: np.ndarray, c: np.ndarray, lamb: float, beta: float, v: callable) -> float:
    return v(c@x + beta).mean() + lamb*norm(x)

def derivative_function_objetive(x: np.ndarray, c):
    pass


# Funciones costo
# Square Loss
@vectorize(list_types_common, nopython=True)
def square_loss(t: float) -> float:
    """
    Calcula el error cuadrático del valor t con respecto al valor de y.

    Args:
        t (float | np.floating)

    Returns: El error cuadrático de la función f con el valor y.

    """
    return (1 - t) ** 2

def derivative_square_loss(t: float):
    return 2*(t-1)

def dderivative():
    pass

# Hinge Loss
@vectorize(list_types_common, nopython=True)
def hinge_loss(t: float) -> float:
    """
    Calcula el error de Hinge de la funcion f con respecto al valor de y.

    Args:
        f (float | np.floating): Predicción del valor.
        y (int | np.integer): Valor real {-1,1}.

    Returns: El error de Hinge de la función f con el valor y.

    """
    return max(1 - t, 0.)


# Smoothed Hinge Loss
@vectorize(list_types_common, nopython=True)
def smooth_hinge_loss(t: float) -> float:
    """
    Calcula el error de Smooth Hinge de la funcion f con respecto al valor de y.

    Args:
        f (float | np.floating): Predicción del valor.
        y (int | np.integer): Valor real {-1,1}.

    Returns: El error de Smooth Hinge de la función f con el valor y.

    """
    if 0 < t <= 1:
        return (1 - t) ** 2 / 2
    return (t <= 0) * (0.5 - t)


# Modified Square Loss
@vectorize(list_types_common, nopython=True)
def mod_square_loss(t: float) -> float:
    """
    Calcula el Modified Square Loss de la funcion f con respecto al valor de y.

    Args:
        f (float | np.floating): Predicción del valor.
        y (int | np.integer): Valor real {-1,1}.

    Returns: El Modified Square Loss de la función f con el valor y.

    """
    return max(1 - t, 0) ** 2


# Exponential Loss
@vectorize(list_types_common, nopython=True)
def exp_loss(t: float) -> float:
    """
    Calcula el Exponential Loss de la funcion f con respecto al valor de y.

    Args:
        f (float | np.floating): Predicción del valor.
        y (int | np.integer): Valor real {-1,1}.

    Returns: El Exponential Loss de la función f con el valor y.

    """
    return math.exp(-t)


# Log loss
@vectorize(list_types_common, nopython=True)
def log_loss(t: float) -> float:
    """
    Calcula la Logistic Loss de la funcion f con respecto al valor de y.

    Args:
        f (float | np.floating): Predicción del valor.
        y (int | np.integer): Valor real {-1,1}.

    Returns: La Logistic Loss de la función f con el valor y.

    """
    return math.log(1 + math.exp(-t))


# Based on Sigmoid Loss
# Falta información de los parametros para el testeo
@vectorize([float64(float64,float64,float64,float64)], nopython=True)
def sigmoid_loss(t: float,
                 gamma: float,
                 theta: float,
                 lamb: float) -> float:
    """
    Calcula la Sigmoid Loss de la funcion f con respecto al valor de y.

    Args:
        f (float | np.floating): Predicción del valor. Debe pertenece el valor entre los valores [-1,1].
        y (int | np.integer): Valor real {-1,1}.
        gamma (float | np.floating): ... Parámetro Gamma del sigmoid.
        theta (float | np.floating): Debe estar entre (0,1). Parámetro Theta del sigmoid.
        lamb (float | np.floating): ... Parámetro Lambda del sigmoid.

    Returns: La Sigmoid Loss de la función f con el valor y.

    """
    if -1 <= t <= 0:
        return (1.2 - gamma) - gamma * t
    elif 0 < t <= theta:
        return (1.2 - lamb) - (1.2 - 2 * gamma) * t / theta
    else:
        return (gamma - lamb * t) / (1 - theta)


# Phi-Learning
@vectorize(list_types_common, nopython=True)
def phi_learning(t: float) -> float:
    """
    Calcula el Phi Learning de la funcion f con respecto al valor de y.

    Args:
        f (float | np.floating): Predicción del valor.
        y (int | np.integer): Valor real {-1,1}.

    Returns: El Phi Learning de la función f con el valor y.

    """
    return 1 - t if 0 <= t <= 1 else 1 - np.sign(t)


# Ramp Loss
@vectorize([float64(float64,float64,float64)], nopython=True)
def ramp_loss(t: float, s: float, c: float) -> float:
    """
    Calcula la Ramp Loss de la funcion f con respecto al valor de y.

    Args:
        f (float | np.floating): Predicción del valor.
        y (int | np.integer): Valor real {-1,1}.
        s (float | np.floating): Valor entre (-1,0].
        c (float | np.floating): Constante.

    Returns: El Ramp Loss de la función f con el valor y.

    """
    return (min(1 - s, max(1 - t, 0)) +
            min(1 - s, max(1 + t, 0)) + c)


# Smooth non-convex loss
@vectorize([float64(float64,float64)], nopython=True)
def smooth_non_convex_loss(t: float, lamb: float) -> float:
    """
    Calcula la Smooth Non Convex Loss de la funcion f con respecto al valor de y.

    Args:
        f (float | np.floating): Predicción del valor.
        y (int | np.integer): Valor real {-1,1}.
        lamb (float | np.floating): Constante.

    Returns: La Smooth Non Convex Loss de la función f con el valor y.

    """
    return 1 - math.tanh(lamb * t)


# 2-layer Neural New-works
@vectorize(list_types_common, nopython=True)
def layer_neural(t: float) -> float:
    """
    Calcula la 2-Layer Nueral New-Works Loss de la funcion f con respecto al valor de y.

    Args:
        f (float | np.floating): Predicción del valor.
        y (int | np.integer): Valor real {-1,1}.

    Returns: La 2-Layer Nueral New-Works Loss de la función f con el valor y.

    """
    return (1 - (1 / (1 + math.exp(-t))))**2


# Logistic difference Loss
@vectorize([float64(float64,float64)], nopython=True)
def logistic_difference_loss(t: float, mu: float) -> float:
    """
    Calcula la Logistic Difference Loss de la funcion f con respecto al valor de y.

    Args:
        f (float | np.floating): Predicción del valor.
        y (int | np.integer): Valor real {-1,1}.
        mu (float | np.floating): Valor controlado.

    Returns: La Logistic Difference Loss de la función f con el valor y.

    """
    return ((math.log(1 + math.exp(-t))) -
            (math.log(1 + math.exp(-t - mu))))


# Smoothed 0-1 Loss
@vectorize(list_types_common, nopython=True)
def smooth01(t: float) -> float:
    """
    Calcula la Smoothed 0-1 Loss de la funcion f con respecto al valor de y.

    Args:
        t (float): Valor de yi(w*xi+b)

    Returns: La Smoothed 0-1 Loss de la función f con el valor y.

    """
    return (t ** 3 - 3 * t + 2) / 4 if -1 <= t <= 1 else t < -1
