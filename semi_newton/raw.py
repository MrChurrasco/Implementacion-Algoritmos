import numpy as np
from scipy.optimize import minimize, Bounds
from numpy.linalg import norm


def semi_newton_paso(fun: callable, grad_fun: callable, bk: callable,
                     xk: np.ndarray[float], beta: float,
                     c: float, t_min: float, t0_bar: float, gamma: float,
                     rho: float, sigma: float) -> np.ndarray[float]:
    wk = grad_fun(*xk)

    # Obteniendo dk
    # Para ello, minimizaremos la norma del vector
    # Obteniendo el vector dk con la dirección de descenso
    # con un vector casi unitario

    # Posicion inicial
    # Cota desigualdad <wk,dk> <=-C||dk||^2
    x0 = np.array(list(-wk / c) + [rho])  # hat{x0} = [x0 rho_max]

    sol = minimize(fun=lambda x: norm(x[:-1]),  # ||d||^2
                   x0=x0,
                   constraints=(
                       {"type": "eq",
                        "fun": lambda x: bk(*x[:-1]) + x[-1] * x[:-1] - wk},  # Bk(dk) + rho_kdk = wk
                       {"type": "ineq",
                        "fun": lambda x: -(np.inner(wk, x[:-1]) + c * norm(x[:-1]))},  # <wk,dk> <=-C||dk||^2
                       {"type": "ineq",
                        "fun": lambda x: norm(x[:-1]) - 1},  # ||d||^2 >= 1
                       {"type": "ineq",
                        "fun": lambda x: x[-1]*(rho-x[-1])}),
                   method='Nelder-Mead',
                   )

    dk = sol.x[:-1]  # Obtenemos el vector dkque cumple con las condiciones

    # Backtracking + trial step
    tau = np.zeros(3)  # tau[i] = tau_(k-i)
    tau_bar = np.zeros(3)  # tau_bar[i] = \bar(tau)_(k-i)

    tau_bar[2] = t0_bar

    # Condición de backtracking: phi(xk + tau_k*dk) > phi(xk)+sigma*tau_k*<wk,dk>
    backtracking_cond = lambda t: fun(*(xk + t * dk)) > fun(*xk) + sigma * t * np.inner(wk, dk)

    # Condiciones iniciales para el loop
    tau[2] = beta * t0_bar if backtracking_cond(t0_bar) else t0_bar
    tau_bar[1] = max([tau[2], t_min])
    tau[1] = beta * tau_bar[1] if backtracking_cond(tau_bar[1]) else tau_bar[1]

    # Ciclo backtracking
    while True:
        # Si tau_(k-2) = tau_bar_(k-2) y tau_(k-1) = tau_bar_(k-1)
        # tau_bar_k = gamma*tau_(k-1)
        # else -> tau_bar_k = max{tau_(k-1), t_min}
        tau_bar[0] = max([tau_bar[1], t_min]) if False in (tau[1:] == tau_bar[1:]) else gamma * tau[1]
        if not backtracking_cond(tau_bar[0]):
            break
        tau[0] = beta * tau_bar[0]  # tau_k = beta*tau_bar_k
        tau[1:] = tau[:2]  # tau_k-1 = tau_k y tau_k-2 = tau_k-1
        tau_bar[1:] = tau_bar[:2]  # tau_bar_k-1 = tau_bar_k y tau_bar_k-2 = tau_bar_k-1

    return xk + tau[0] * dk


if __name__ == '__main__':
    semi_newton_paso(lambda x, y: x ** 2 + y ** 2,
                     lambda x, y: np.array([2 * x, 2 * y]),
                     lambda x, y: np.array([[2, 0], [0, 2]]),
                     np.array([1., -1.]),
                     0.5,
                     0.5,
                     0.5,
                     1,
                     1.1,
                     2,
                     2)
