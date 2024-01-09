import numpy as np
from numpy.linalg import norm

def backtracking(fun, grad_fun, xk: np.ndarray[float], pk: np.ndarray, alpha: float = 1, beta: float = 0.5,
                 c: float = 1e-4) -> float:
    """Método para calcular $\alpha$, usando las condiciones de Goldstein y el método de backtracking.

    Args:
        fun (function): Función por la cual se quiere descender.
        xk (np.ndarray[float]): Valor de referencia para hacer el descenso.
        grad_fun (function): El gradiente de la función 'fun'.
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


def alg1(wk, bk, c, rho_max):
    """
    while -c * norm(dk) < np.inner(wk, dk):
        pk = np.random.rand(1) * rho  # Choose [0, rho_max]
        dk = np.random.rand(x0.shape)
        wk = -bk(dk) - pk * dk
    """
    return 1


def trial_step(fun, grad_fun, xk, dk, t0_bar: float, gamma: float,
               t_min: float, beta, sigma):

    t0 = backtracking(fun, grad_fun, xk, dk, t0_bar, beta, sigma)
    t1_bar = max(t0, t_min)
    t1 = backtracking(fun, grad_fun, xk, dk, t1_bar, beta, sigma)
    t2_bar = 0
    while True:
        if t0 == t0_bar and t1 == t1_bar:
            t2_bar = gamma*t1
        else:
            t2_bar = max(t1, t_min)
        t2 = backtracking(fun, grad_fun, xk, dk, t2_bar, beta, sigma)
        if t2 == t2_bar:
            break
        t0_bar = t1_bar
        t0 = t1
        t1_bar = t2_bar
        t1 = t2
    return t2


def semi_newton_paso(fun, grad_fun, bk, xk: np.ndarray[float], beta: float,
                     c: float, t_min: float, t0_bar: float, gamma: float,
                     rho: float, sigma: float) -> np.ndarray[float]:
    wk = grad_fun(xk)
    d_k = alg1(wk, bk, c, rho)
    t_k = trial_step(t0_bar=t0_bar,
                     gamma=gamma,
                     t_min=t_min,
                     dk=d_k,
                     fun=fun,
                     grad_fun=grad_fun,
                     xk=xk,
                     beta=beta,
                     sigma=sigma
                     )

    return xk + t_k * d_k
