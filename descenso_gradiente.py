"""
@author: Sebastian Carrasco
"""

import numpy as np
from numpy.linalg import inv


def descenso_gradiente(
        fun,
        xk: np.ndarray[float],
        grad_fun,
        hessian_fun=None,
        method: str = "step",
) -> np.ndarray[float]:
    """Algoritmo del descenso del gradiente implementado con numpy.
    Acepta funciones, gradiente de funciones y hessianos que acepten un vector de numpy (ndarray).
    Están disponibles 3 métodos: step, newton y qnewton.

    Args:
        fun (function): Función por la cual se quiere descender.
        xk (np.ndarray[float]): Valor de referencia para hacer el descenso.
        grad_fun (function): El gradiente de la función 'fun'.
        hessian_fun (function, optional): El Hessiano de la función 'fun'. Defaults to None.
        method (str, optional): Método que se utiliza para hacer el calculo del descenso. Defaults to "step".

    Raises:
        ValueError: Escribir el nombre de un método no implementado.
        ValueError: Usar el método de newton o qnewton sin entregar el hessiano de la función.

    Returns:
        np.ndarray[float]: Retorna el siguiente valor del descenso.
    """
    # Casos borde
    # Escoger un metodo que no existe
    if method not in ["step", "newton", "qnewton"]:
        raise ValueError(
            f"{method} no es uno de los métodos implementados. Utilice steep, newton o qnewton."
        )
    # Escoger algun metodo de newton y no tener el hessiano de la funcion
    if (hessian_fun is None) and (method in ["newton", "qnewton"]):
        raise ValueError(
            f"Para utilizar el método de newton o quasi-newton se requiere el hessiano de la funcion."
        )

    # Implementación

    # Calculo p_k
    else:
        if method == "step":
            bk = np.identity(xk.shape[0])
        else:
            bk = hessian_fun(xk)

        pk: np.ndarray[float] = -inv(bk) @ grad_fun(xk)

    # Calculo alpha_k

    def backtracking(
            fun,
            grad_fun,
            xk: np.ndarray[float],
            pk: np.ndarray,
            alpha: float = 1,
            beta: float = 0.5,
            c: float = 1e-4,
    ) -> float:
        """Método para calcular $\alpha$, usando las condiciones de Goldstein y el método de backtracking.

        Args:
            pk (np.ndarray): Vector que índica la dirección de descenso.
            alpha (float, optional): Ponderador del vector pk. Defaults to 1.
            beta (float, optional): Tasa de disminución del valor de alpha al no cumplir las condiciones de Goldstein.
            Defaults to 0.5.
            c (float, optional): Constante entre (0,1). Defaults to 1e-4.

        Returns:
            float: Retorna el primer $\alpha$ encontrado que cumple las condiciones de Goldstein.
        """
        # Calculamos las constantes del ciclo antes (Optimización)
        rhs = c * grad_fun(xk) @ pk
        fxk = fun(xk)
        while fun(xk + alpha * pk) > fxk + alpha * rhs:
            alpha = beta * alpha
        return alpha

    alpha_k: float = backtracking(fun, grad_fun, xk, pk)

    # Retornamos el x_(k+1)
    return xk + alpha_k * pk


def main():
    print(list(map(lambda x: x + 2, (2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14))))


if __name__ == '__main__':
    main()
