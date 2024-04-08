import numpy as np
from numpy.linalg import norm, solve
from scipy.optimize import minimize_scalar


def backtracking(fun: callable, wk: np.ndarray, xk: np.ndarray, dk: np.ndarray, tk: np.ndarray | float = 1,
                 beta: float = 0.5, sigma: float = 1e-4) -> float:
    """Método para calcular alpha, usando las condiciones de Goldstein y el método de backtracking.

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
        float: Retorna el primer alpha encontrado que cumple las condiciones de Goldstein.
    """
    # Calculamos las constantes del ciclo antes (Optimización)
    rhs = sigma * wk @ dk
    fxk = fun(*xk)
    while fun(*(xk + tk * dk)) > fxk + tk * rhs:
        tk = beta * tk
    return tk


def semi_newton_paso(fun: callable, grad_fun: callable, bk: np.ndarray,
                     xk: np.ndarray, beta: float,
                     c: float, t_min: float, gamma: float,
                     rho: float, sigma: float, num_paso: int,
                     tau_hist: np.ndarray, tau_bar_hist: np.ndarray,
                     rho_opt: bool = False) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    wk = grad_fun(*xk)

    # Obteniendo dk
    if rho_opt:
        # Para ello, maximizaremos -<wk,dk> -C||dk||^2
        # variando los posibles valores para rho
        # Obteniendo el vector dk con la dirección de descenso
        def objetivo(x):
            y = solve(bk + np.identity(bk.shape[0]) * x, -wk)  # Bk(dk) + rho_kdk = wk
            return -(np.inner(wk, y) + c * norm(y))  # <wk,dk> <=-C||dk||^2

        # Buscamos el mejor dk y rho
        sol = minimize_scalar(fun=objetivo, bounds=(0, rho))

        # Obtenemos el vector dk que cumple con las condiciones
        dk: np.ndarray = solve(bk + np.identity(bk.shape[0]) * sol.x, -wk)

    else:
        while True:
            dk_sol = solve(bk + np.identity(bk.shape[0]) * rho, -wk)
            if np.inner(wk, dk_sol) <= -c * norm(dk_sol):
                break
            rho = rho / (1 >> 1)

        dk: np.ndarray = solve(bk + np.identity(bk.shape[0]) * rho, -wk)

    # Backtracking + Self-adaptive trial stepsize
    if num_paso == 0:
        tau_hist[2] = backtracking(fun=fun, wk=wk, xk=xk, dk=dk, tk=tau_bar_hist[2], beta=beta, sigma=sigma)
        return xk + tau_hist[2] * dk, tau_hist, tau_bar_hist
    elif num_paso == 1:
        tau_bar_hist[1] = max([tau_hist[2], t_min])
        tau_hist[1] = backtracking(fun=fun, wk=wk, xk=xk, dk=dk, tk=tau_bar_hist[1], beta=beta, sigma=sigma)
        return xk + tau_hist[1] * dk, tau_hist, tau_bar_hist
    else:
        # Si tau_(k-2) = tau_bar_(k-2) y tau_(k-1) = tau_bar_(k-1)
        if (tau_hist[1] == tau_bar_hist[1]) and (tau_hist[2] == tau_bar_hist[2]):
            tau_bar_hist[0] = gamma * tau_hist[1]
        # else -> tau_bar_k = max{tau_(k-1), t_min}
        else:
            tau_bar_hist[0] = max([tau_hist[1], t_min])

        tau_hist[0] = backtracking(fun=fun, wk=wk, xk=xk, dk=dk, tk=tau_bar_hist[0], beta=beta, sigma=sigma)
        return xk + tau_hist[0] * dk, tau_hist, tau_bar_hist


def semi_newton(fun: callable, grad_fun: callable, bk: np.ndarray, xk: np.ndarray, beta: float,
                c: float, t_min: float, t0_bar: float, gamma: float,
                rho: float, sigma: float, tol: float = 1e-4) -> tuple:
    k = 0
    tau = np.zeros(3)  # tau[i] = tau_(k-i)
    tau_bar = np.zeros(3)  # tau_bar[i] = \bar(tau)_(k-i)

    tau_bar[2] = t0_bar
    print(xk, tau, tau_bar, k)
    while True:
        xk1, tau, tau_bar = semi_newton_paso(fun=fun, grad_fun=grad_fun, bk=bk, xk=xk, beta=beta,
                                             c=c, t_min=t_min, gamma=gamma, rho=rho, sigma=sigma,
                                             tau_hist=tau, tau_bar_hist=tau_bar, rho_opt=False, num_paso=k)
        k += 1
        if norm(grad_fun(*xk)) <= tol:
            break
        xk = xk1
        if k > 2:
            tau[2], tau_bar[2] = tau[1], tau_bar[1]
            tau[1], tau_bar[1] = tau[0], tau_bar[0]
        print(xk, tau, tau_bar, k)

    return xk, k, norm(grad_fun(*xk))


def main():
    a1 = semi_newton(fun=lambda x, y: x ** 2 + y ** 2,
                     grad_fun=lambda x, y: np.array([2 * x, 2 * y]),
                     bk=np.array([[2, 0], [0, 2]]),
                     xk=np.array([1., -1.]),
                     beta=0.5,
                     c=0.5,
                     t_min=0.4,
                     t0_bar=1,
                     gamma=1.1,
                     rho=2,
                     sigma=2,
                     tol=1e-40)

    print(a1)


if __name__ == '__main__':
    main()
