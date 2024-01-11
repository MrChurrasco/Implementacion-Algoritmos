import numpy as np
from scipy.optimize import minimize_scalar
from numpy.linalg import norm, solve


def semi_newton_paso(fun: callable, grad_fun: callable, bk: np.ndarray[float],
                     xk: np.ndarray[float], beta: float,
                     c: float, t_min: float, t0_bar: float, gamma: float,
                     rho: float, sigma: float) -> np.ndarray[float]:

    wk = grad_fun(*xk)

    # Obteniendo dk
    # Para ello, maximizaremos -<wk,dk> -C||dk||^2
    # variando los posibles valores para rho
    # Obteniendo el vector dk con la dirección de descenso
    def objetivo(x):
        dk_sol = solve(bk + np.identity(bk.shape[0])*x, -wk)  # Bk(dk) + rho_kdk = wk
        return -(np.inner(wk, dk_sol) + c * norm(dk_sol))  # <wk,dk> <=-C||dk||^2

    # Buscamos el mejor dk y rho
    sol = minimize_scalar(fun=objetivo, bounds=(0, rho))

    # Obtenemos el vector dk que cumple con las condiciones
    dk = solve(bk + np.identity(bk.shape[0])*sol.x, -wk)

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
        tau_bar[0] = gamma * tau[1] if (tau[1] == tau_bar[1]) and (tau[2] == tau_bar[2]) else max([tau_bar[1], t_min])
        tau[0] = tau_bar[0]
        if not backtracking_cond(tau[0]):
            break
        tau[0] = beta * tau[0]  # tau_k = beta*tau_k
        tau[1], tau[2] = tau[0], tau[1]  # tau_k-1 = tau_k y tau_k-2 = tau_k-1
        tau_bar[1], tau_bar[2] = tau_bar[0], tau_bar[1]  # tau_bar_k-1 = tau_bar_k y tau_bar_k-2 = tau_bar_k-1

    return xk + tau[0] * dk


if __name__ == '__main__':
    xk = semi_newton_paso(fun=lambda x, y: x ** 2 + y ** 2,
                          grad_fun=lambda x, y: np.array([2 * x, 2 * y]),
                          bk=np.array([[2, 0], [0, 2]]),
                          xk=np.array([1., -1.]),
                          beta=0.5,
                          c=0.5,
                          t_min=0.4,
                          t0_bar=1,
                          gamma=1.1,
                          rho=2,
                          sigma=2)

    print(xk)
