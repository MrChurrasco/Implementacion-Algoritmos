from typing import Tuple, Any

import numpy as np
from numpy import ndarray
from scipy.optimize import minimize_scalar
from numpy.linalg import norm, solve


def backtracking(fun, wk: np.ndarray, xk: np.ndarray, dk: np.ndarray, tk: np.ndarray | float = 1, beta: float = 0.5,
                 sigma: float = 1e-4) -> float:
    """Método para calcular $\alpha$, usando las condiciones de Goldstein y el método de backtracking.

    Args:
        fun (function): Función por la cual se quiere descender.
        xk (np.ndarray[float]): Valor de referencia para hacer el descenso.
        wk (function): El gradiente de la función 'fun'.
        dk (np.ndarray): Vector que índica la dirección de descenso.
        tk (float, optional): Ponderador del vector pk. Defaults to 1.
        beta (float, optional): Tasa de disminución del valor de alpha al no cumplir las condiciones de Goldstein.
        Defaults to 0.5.
        sigma (float, optional): Constante entre (0,1). Defaults to 1e-4.

    Returns:
        float: Retorna el primer $\alpha$ encontrado que cumple las condiciones de Goldstein.
    """
    # Calculamos las constantes del ciclo antes (Optimización)
    rhs = sigma * wk @ dk
    fxk = fun(*xk)
    while fun(*(xk + tk * dk)) > fxk + tk * rhs:
        tk = beta * tk
    return tk


def semi_newton_paso(fun: callable, grad_fun: callable, bk: np.ndarray,
                     xk: np.ndarray, beta: float,
                     c: float, t_min: float, t0_bar: float, gamma: float,
                     rho: float, sigma: float) -> np.ndarray:
    wk = grad_fun(*xk)

    # Obteniendo dk
    # Para ello, maximizaremos -<wk,dk> -C||dk||^2
    # variando los posibles valores para rho
    # Obteniendo el vector dk con la dirección de descenso
    def objetivo(x):
        dk_sol = solve(bk + np.identity(bk.shape[0]) * x, -wk)  # Bk(dk) + rho_kdk = wk
        return -(np.inner(wk, dk_sol) + c * norm(dk_sol))  # <wk,dk> <=-C||dk||^2

    # Buscamos el mejor dk y rho
    sol = minimize_scalar(fun=objetivo, bounds=(0, rho))

    # Obtenemos el vector dk que cumple con las condiciones
    dk: np.ndarray = solve(bk + np.identity(bk.shape[0]) * sol.x, -wk)

    # Backtracking + trial step
    tau = np.zeros(3)  # tau[i] = tau_(k-i)
    tau_bar = np.zeros(3)  # tau_bar[i] = \bar(tau)_(k-i)

    tau_bar[2] = t0_bar

    # Condiciones iniciales para el loop
    tau[2] = backtracking(fun=fun, wk=wk, xk=xk, dk=dk, tk=t0_bar, beta=beta, sigma=sigma)
    tau_bar[1] = max([tau[2], t_min])
    tau[1] = backtracking(fun=fun, wk=wk, xk=xk, dk=dk, tk=tau_bar[1], beta=beta, sigma=sigma)

    # Ciclo backtracking
    while True:
        # Si tau_(k-2) = tau_bar_(k-2) y tau_(k-1) = tau_bar_(k-1)
        # tau_bar_k = gamma*tau_(k-1)
        # else -> tau_bar_k = max{tau_(k-1), t_min}
        tau_bar[0] = gamma * tau[1] if (tau[1] == tau_bar[1]) and (tau[2] == tau_bar[2]) else max([tau[1], t_min])
        tau[0] = tau_bar[0]
        if fun(*(xk + tau[0] * dk)) > fun(*xk) + tau[0] * sigma * wk @ dk:
            break
        tau[0] = beta * tau[0]  # tau_k = beta*tau_k
        tau[1], tau[2] = tau[0], tau[1]  # tau_k-1 = tau_k y tau_k-2 = tau_k-1
        tau_bar[1], tau_bar[2] = tau_bar[0], tau_bar[1]  # tau_bar_k-1 = tau_bar_k y tau_bar_k-2 = tau_bar_k-1

    return xk + tau[0] * dk


def semi_newton(fun: callable, grad_fun: callable, bk: np.ndarray, x0: np.ndarray, beta: float,
                c: float, t_min: float, t0_bar: float, gamma: float,
                rho: float, sigma: float, tol: float = 1e-4) -> tuple[ndarray, int, Any]:
    xk = semi_newton_paso(fun=fun, grad_fun=grad_fun, bk=bk, xk=x0, beta=beta,
                          c=c, t_min=t_min, t0_bar=t0_bar, gamma=gamma, rho=rho, sigma=sigma)

    k = 1
    while np.sqrt(norm(xk - x0)) > tol:
        x0 = xk
        xk = semi_newton_paso(fun=fun, grad_fun=grad_fun, bk=bk, xk=x0, beta=beta,
                              c=c, t_min=t_min, t0_bar=t0_bar, gamma=gamma, rho=rho, sigma=sigma)
        k += 1
    return xk, k, norm(grad_fun(*xk))


if __name__ == '__main__':
    a1 = semi_newton(fun=lambda x, y: x ** 2 + y ** 2,
                     grad_fun=lambda x, y: np.array([2 * x, 2 * y]),
                     bk=np.array([[2, 0], [0, 2]]),
                     x0=np.array([1., -1.]),
                     beta=0.5,
                     c=0.5,
                     t_min=0.4,
                     t0_bar=1,
                     gamma=1.1,
                     rho=2,
                     sigma=2,
                     tol=1e-40)

    print(a1)
