import math
import numpy as np


# Funciones costo
# Square Loss
def square_loss(f: float | np.floating,
                y: int | np.integer) -> float:
    """
    Calcula el error cuadrático de la funcion f con respecto al valor de y.

    Args:
        f (float | np.floating): Predicción del valor.
        y (int | np.integer): Valor real {-1,1}.

    Returns: El error cuadrático de la función f con el valor y.

    """
    return math.pow((y - f), 2)


# Hinge Loss
def hinge_loss(f: float | np.floating,
               y: int | np.integer) -> float | np.floating:
    """
    Calcula el error de Hinge de la funcion f con respecto al valor de y.

    Args:
        f (float | np.floating): Predicción del valor.
        y (int | np.integer): Valor real {-1,1}.

    Returns: El error de Hinge de la función f con el valor y.

    """
    return max(1 - y * f, 0.)


# Smoothed Hinge Loss
def smooth_hinge_loss(f: float | np.floating,
                      y: int | np.integer) -> float | np.floating:
    """
    Calcula el error de Smooth Hinge de la funcion f con respecto al valor de y.

    Args:
        f (float | np.floating): Predicción del valor.
        y (int | np.integer): Valor real {-1,1}.

    Returns: El error de Smooth Hinge de la función f con el valor y.

    """
    a = y * f
    return math.pow((1 - a), 2) / 2 if 0 < a <= 1 else (a <= 0) * (0.5 - a)


# Modified Square Loss
def mod_square_loss(f: float | np.floating,
                    y: int | np.integer) -> float | np.floating:
    """
    Calcula el Modified Square Loss de la funcion f con respecto al valor de y.

    Args:
        f (float | np.floating): Predicción del valor.
        y (int | np.integer): Valor real {-1,1}.

    Returns: El Modified Square Loss de la función f con el valor y.

    """
    return math.pow(max(1 - y * f, 0), 2)


# Exponential Loss
def exp_loss(f: float | np.floating,
             y: int | np.integer) -> float | np.floating:
    """
    Calcula el Exponential Loss de la funcion f con respecto al valor de y.

    Args:
        f (float | np.floating): Predicción del valor.
        y (int | np.integer): Valor real {-1,1}.

    Returns: El Exponential Loss de la función f con el valor y.

    """
    return math.exp(-y * f)


# Log loss
def log_loss(f: float | np.floating,
             y: int | np.integer) -> float | np.floating:
    """
    Calcula la Logistic Loss de la funcion f con respecto al valor de y.

    Args:
        f (float | np.floating): Predicción del valor.
        y (int | np.integer): Valor real {-1,1}.

    Returns: La Logistic Loss de la función f con el valor y.

    """
    return math.log(1 + math.exp(-y * f))


# Based on Sigmoid Loss
# Falta información de los parametros para el testeo
def sigmoid_loss(f: float | np.floating,
                 y: int | np.integer,
                 gamma: float | np.floating,
                 theta: float | np.floating,
                 lamb: float | np.floating) -> float | np.floating | None:
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
    a = y * f
    if -1 <= a <= 0:
        return (1.2 - gamma) - gamma * a
    if 0 < a <= theta:
        return (1.2 - lamb) - (1.2 - 2 * gamma) * a / theta
    if theta < a <= 1:
        return (gamma - lamb * a) / (1 - theta)
    return None


# Phi-Learning
def phi_learning(f: float | np.floating,
                 y: int | np.integer) -> float | np.floating:
    """
    Calcula el Phi Learning de la funcion f con respecto al valor de y.

    Args:
        f (float | np.floating): Predicción del valor.
        y (int | np.integer): Valor real {-1,1}.

    Returns: El Phi Learning  de la función f con el valor y.

    """
    a = y * f
    return 1 - a if 0 <= a <= 1 else 1 - np.sign(a)


# Ramp Loss
def ramp_loss(f: float | np.floating,
              y: int | np.integer,
              s: float | np.floating,
              c: float | np.floating) -> float | np.floating:
    """
    Calcula la Ramp Loss de la funcion f con respecto al valor de y.

    Args:
        f (float | np.floating): Predicción del valor.
        y (int | np.integer): Valor real {-1,1}.
        s (float | np.floating): Valor entre (-1,0].
        c (float | np.floating): Constante.

    Returns: El Ramp Loss de la función f con el valor y.

    """
    return (min(1 - s, max(1 - f * y, 0)) +
            min(1 - s, max(1 + f * y, 0)) + c)


# Smooth non-convex loss
def smooth_non_convex_loss(f: float | np.floating,
                           y: int | np.integer,
                           lamb: float | np.floating) -> float | np.floating:
    """
    Calcula la Smooth Non Convex Loss de la funcion f con respecto al valor de y.

    Args:
        f (float | np.floating): Predicción del valor.
        y (int | np.integer): Valor real {-1,1}.
        lamb (float | np.floating): Constante.

    Returns: La Smooth Non Convex Loss de la función f con el valor y.

    """
    return 1 - math.tanh(lamb * y * f)


# 2-layer Neural New-works
def layer_neural(f: float | np.floating,
                 y: int | np.integer) -> float | np.floating:
    """
    Calcula la 2-Layer Nueral New-Works Loss de la funcion f con respecto al valor de y.

    Args:
        f (float | np.floating): Predicción del valor.
        y (int | np.integer): Valor real {-1,1}.

    Returns: La 2-Layer Nueral New-Works Loss de la función f con el valor y.

    """
    return math.pow((1 - (1 / (1 + math.exp(-y * f)))), 2)


# Logistic difference Loss
def logistic_difference_loss(f: float | np.floating,
                             y: int | np.integer,
                             mu: float | np.floating) -> float | np.floating:
    """
    Calcula la Logistic Difference Loss de la funcion f con respecto al valor de y.

    Args:
        f (float | np.floating): Predicción del valor.
        y (int | np.integer): Valor real {-1,1}.
        mu (float | np.floating): Valor controlado.

    Returns: La Logistic Difference Loss de la función f con el valor y.

    """
    return ((math.log(1 + math.exp(-y * f))) -
            (math.log(1 + math.exp(-y * f - mu))))


# Smoothed 0-1 Loss
def smooth01(t: float | np.floating) -> float | np.floating:
    """
    Calcula la Smoothed 0-1 Loss de la funcion f con respecto al valor de y.

    Args:
        t (float | np.floating): Valor de yi(w*xi+b)

    Returns: La Smoothed 0-1 Loss de la función f con el valor y.

    """
    return (math.pow(t, 3) - 3 * t + 2) / 4 if -1 <= t <= 1 else t < -1
