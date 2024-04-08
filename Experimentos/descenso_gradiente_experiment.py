"""
@author: Sebastian Carrasco
@anno: 2024
"""
from API_Experiments.api import Algorithm
from typing import Any

import numpy as np
from numpy.linalg import norm
from time import perf_counter_ns


def backtracking(fun, grad_fun, xk: np.ndarray[float], pk: np.ndarray[float],
                 alpha: float, beta: float, c: float) -> float:
    """Método para calcular alpha, usando las condiciones de Goldstein y el método de backtracking.

    Args:
        fun (function): Función por la cual se quiere descender.
        xk (np.ndarray[float]): Valor de referencia para hacer el descenso.
        grad_fun (function): El gradiente de la función 'fun'.
        pk (np.ndarray): Vector que índica la dirección de descenso.
        alpha (float): Ponderador del vector pk.
        beta (float): Tasa de disminución del valor de alpha al no cumplir las condiciones de Goldstein.
        c (float, optional): Constante entre (0,1).

    Returns:
        float: Retorna el primer alpha encontrado que cumple las condiciones de Goldstein.
    """
    # Calculamos las constantes del ciclo antes (Optimización)
    rhs = c * grad_fun(*xk) @ pk
    fxk = fun(*xk)
    while fun(*(xk + alpha * pk)) > fxk + alpha * rhs:
        alpha = beta * alpha
    return alpha


def descenso_gradiente_paso(fun, grad_fun, xk: np.ndarray[float],
                            alpha: float = 1, beta: float = 0.5, c: float = 1e-4) -> np.ndarray[float]:
    """Paso del algoritmo del descenso del gradiente implementado con numpy.
    Acepta funciones, gradiente de funciones y hessianos que acepten un vector de numpy (ndarray).
    Están disponibles 3 métodos: step, newton y qnewton.

    Args:
        fun (function): Función por la cual se quiere descender.
        xk (np.ndarray[float]): Valor de referencia para hacer el descenso.
        grad_fun (function): El gradiente de la función 'fun'.
        alpha (float, optional): Ponderador del vector pk. Defaults to 1.
        beta (float, optional): Tasa de disminución del valor de alpha al no cumplir las condiciones de Goldstein.
        Defaults to 0.5.
        c (float, optional): Constante entre (0,1). Defaults to 1e-4.

    Returns:
        np.ndarray[float]: Retorna el siguiente valor del descenso.
    """

    # Implementación
    pk: np.ndarray[float] = -np.identity(xk.shape[0]) @ grad_fun(*xk)

    # Calculo alpha_k
    alpha_k: float = backtracking(fun, grad_fun, xk, pk, alpha, beta, c)

    # Retornamos el x_(k+1)
    return xk + alpha_k * pk


class GradientDescent(Algorithm):

    def __init__(self):
        name: str = "Gradient Descent"
        inputs: dict[str, type] = {
            "fun": callable,
            "grad_fun": callable,
            "xk": np.ndarray,
            "alpha": float,
            "beta": float,
            "c": float,
            "tol": float
        }
        outputs: dict[str, type] = {
            "xk": np.ndarray,
            "num_iter": int,
            "norm_grad_fun": float,
            "t_iter": int
        }
        super().__init__(name, inputs, outputs)

    def run_algorithm(self, **kwargs) -> dict[str, Any]:
        # Extraemos las condiciones iniciales del diccionario
        fun = kwargs.get("fun")
        grad_fun = kwargs.get("grad_fun")
        xk: np.ndarray[float] = kwargs.get("xk")
        alpha: float = kwargs.get("alpha")
        beta: float = kwargs.get("beta")
        c: float = kwargs.get("c")
        tol: float = kwargs.get("tol", 1e-40)

        # Procedemos a escribir el algoritmo
        k = 0

        # Corremos el algoritmo de Descenso del Gradiente por paso
        while norm(grad_fun(*xk)) >= tol:
            # Aplicamos las condiciones de paso
            t_ini = perf_counter_ns()
            xk1 = descenso_gradiente_paso(fun=fun,
                                          grad_fun=grad_fun,
                                          xk=xk,
                                          alpha=alpha,
                                          beta=beta,
                                          c=c)
            t_fin = perf_counter_ns() - t_ini
            k += 1
            xk = xk1

            # Guardamos el output de cada iteración del algoritmo
            self.save_output({"xk": xk,
                              "num_iter": k,
                              "norm_grad_fun": norm(grad_fun(*xk)),
                              "t_iter": t_fin
                              })

        # Retornamos el output final en forma de diccionario
        return {"xk": xk,
                "num_iter": k,
                "norm_grad_fun": norm(grad_fun(*xk))
                }
