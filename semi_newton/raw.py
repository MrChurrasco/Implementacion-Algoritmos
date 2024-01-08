import numpy as np
from numpy.linalg import norm

def semi_newton(fun: function, grad_fun: function, bk, x0: np.ndarray[float], beta: float, 
                c: float, t_min: float, t0: float, gamma: float,
                rho: float, sigma: float, tol: float = 1e-10)-> np.ndarray[float]:
    xk = x0
    while True:
        wk = grad_fun(x0)
        if wk == 0:
            break
        
        while -c*norm(dk) < np.inner(wk,dk):
            pk = np.random.rand(1)*rho # Choose [0, rho_max]
            dk = np.random.rand(x0.shape)
            wk = -bk(dk) - pk*dk
        
        tk = trial_step(t0,gamma,t_min)
        
        while fun(xk + tk*dk) > fun(xk) + sigma*tk*np.inner(wk,dk):
            tk *= beta
        xk += tk*dk
    return xk    



def trial_step(t0: float, gamma: float, t_min: float):
    t1_bar = max(t0, t_min)
    while True:
        if tk2 == tk2_bar and tk1 == tk1_bar:
            tk_bar = gamma*tk1
        else:
            tk_bar = max[tk1, t_min]
    pass
