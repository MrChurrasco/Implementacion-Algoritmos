import math
import numpy as np


# Funciones costo
# Square Loss
def square_loss(f: float, y: int) -> float:
    """
    Calcula el error cuadrático de la funcion f con respecto al valor de y.

    Args:
        f (float): Predicción del valor.
        y (int): Valor real {-1,1}.

    Returns: El error cuadrático de la función f con el valor y.

    """
    return math.pow((y - f), 2)


# Hinge Loss
def hinge_loss(f: float, y: int) -> float:
    """
    Calcula el error de Hinge de la funcion f con respecto al valor de y.

    Args:
        f (float): Predicción del valor.
        y (int): Valor real {-1,1}.

    Returns: El error de Hinge de la función f con el valor y.

    """
    return max(1 - y * f, 0)


# Smoothed Hinge Loss
def smooth_hinge_loss(f: float, y: int) -> float:
    """
    Calcula el error de Smooth Hinge de la funcion f con respecto al valor de y.

    Args:
        f (float): Predicción del valor.
        y (int): Valor real {-1,1}.

    Returns: El error de Smooth Hinge de la función f con el valor y.

    """
    a = y * f
    if 0 < a <= 1:
        return math.pow((1 - a), 2) / 2
    return (a <= 0) * (0.5 - a)


# Modified Square Loss
def mod_square_loss(f: float, y: int) -> float:
    """
    Calcula el Modified Square Loss de la funcion f con respecto al valor de y.

    Args:
        f (float): Predicción del valor.
        y (int): Valor real {-1,1}.

    Returns: El Modified Square Loss de la función f con el valor y.

    """
    return math.pow(max(1 - y * f, 0), 2)


# Exponential Loss
def exp_loss(f: float, y: int) -> float:
    """
    Calcula el Exponential Loss de la funcion f con respecto al valor de y.

    Args:
        f (float): Predicción del valor.
        y (int): Valor real {-1,1}.

    Returns: El Exponential Loss de la función f con el valor y.

    """
    return math.exp(-y * f)


# Log loss
def log_loss(f: float, y: int) -> float:
    """
    Calcula la Logistic Loss de la funcion f con respecto al valor de y.

    Args:
        f (float): Predicción del valor.
        y (int): Valor real {-1,1}.

    Returns: La Logistic Loss de la función f con el valor y.

    """
    return math.log(1 + math.exp(-y * f))


# Based on Sigmoid Loss
def sigmoid_loss(f: float, y: int, gamma: float, theta: float, lamb: float) -> float:
    """
    Calcula la Sigmoid Loss de la funcion f con respecto al valor de y.

    Args:
        f (float): Predicción del valor.
        y (int): Valor real {-1,1}.
        gamma (float): ... Parámetro Gamma del sigmoid.
        theta (float): Debe estar entre (0,1). Parámetro Theta del sigmoid.
        lamb (float): ... Parámetro Lambda del sigmoid.

    Returns: La Sigmoid Loss de la función f con el valor y.

    """
    a = y * f
    if -1 <= a <= 0:
        return (1.2 - gamma) - gamma * a
    elif 0 < a <= theta:
        return (1.2 - lamb) - (1.2 - 2 * gamma) * a / theta
    elif theta < a <= 1:
        return (gamma - lamb * a) / (1 - theta)


# Phi-Learning
def phi_learning(f: float, y: int) -> float:
    """
    Calcula el Phi Learning de la funcion f con respecto al valor de y.

    Args:
        f (float): Predicción del valor.
        y (int): Valor real {-1,1}.

    Returns: El Phi Learning  de la función f con el valor y.

    """
    a = y * f
    if 0 <= a <= 1:
        return 1 - a
    return 1 - np.sign(a)


# Ramp Loss
def ramp_loss(f: float, y: int, s: float, c: float) -> float:
    """
    Calcula la Ramp Loss de la funcion f con respecto al valor de y.

    Args:
        f (float): Predicción del valor.
        y (int): Valor real {-1,1}.
        s (float): Valor entre (-1,0].
        c (float): Constante.

    Returns: El Ramp Loss de la función f con el valor y.

    """
    return (min(1 - s, max(1 - f * y, 0)) +
            min(1 - s, max(1 + f * y, 0)) + c)


# Smooth non-convex loss
def smooth_non_convex_loss(f: float, y: int, lamb: float) -> float:
    """
    Calcula la Smooth Non Convex Loss de la funcion f con respecto al valor de y.

    Args:
        f (float): Predicción del valor.
        y (int): Valor real {-1,1}.
        lamb (float): Constante.

    Returns: La Smooth Non Convex Loss de la función f con el valor y.

    """
    return 1 - math.tanh(lamb * y * f)


# 2-layer Neural New-works
def layer_neural(f: float, y: int) -> float:
    """
    Calcula la 2-Layer Nueral New-Works Loss de la funcion f con respecto al valor de y.

    Args:
        f (float): Predicción del valor.
        y (int): Valor real {-1,1}.

    Returns: La 2-Layer Nueral New-Works Loss de la función f con el valor y.

    """
    return math.pow((1 - (1 / (1 + math.exp(-y * f)))), 2)


# Logistic difference Loss
def logistic(f: float, y: float, mu: float) -> float:
    """
    Calcula la Logistic Difference Loss de la funcion f con respecto al valor de y.

    Args:
        f (float): Predicción del valor.
        y (int): Valor real {-1,1}.
        mu (float): Valor controlado.

    Returns: La Logistic Difference Loss de la función f con el valor y.

    """
    return ((math.log(1 + math.exp(-y * f))) -
            (math.log(1 + math.exp(-y * f - mu))))


# Smoothed 0-1 Loss
def smooth01(t: float) -> float:
    """
    Calcula la Smoothed 0-1 Loss de la funcion f con respecto al valor de y.

    Args:
        t (float): Valor de yi(w*xi+b)

    Returns: La Smoothed 0-1 Loss de la función f con el valor y.

    """
    if -1 <= t <= 1:
        return (math.pow(t, 3) - 3 * t + 2) / 4
    return t < -1
