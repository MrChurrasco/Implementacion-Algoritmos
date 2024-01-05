import funciones_costo as fc
import pytest
import numpy as np

##
clases = [-1, 1]
rango_test = np.linspace(-5, 5, 20, dtype=np.float16)


# Square Loss
@pytest.mark.parametrize(
    "fx_test, y_test",
    [(fx, y) for fx in rango_test
     for y in [-1, 1]]
)
def test_square_loss_positive_always(fx_test, y_test):
    assert fc.square_loss(fx_test, y_test) > 0


@pytest.mark.parametrize(
    "fx_test, y_test, expected",
    [(fx, y, (1 - fx * y) ** 2)
     for fx in rango_test
     for y in clases]
)
def test_square_loss(fx_test, y_test, expected):
    assert fc.square_loss(fx_test, y_test) == expected


# Hinge Loss
@pytest.mark.parametrize(
    "fx_test, y_test, expected",
    [(fx, y, 1 - fx * y) if fx * y < 1
     else (fx, y, 0)
     for fx in rango_test
     for y in clases]
)
def test_hinge_loss(fx_test, y_test, expected):
    assert fc.hinge_loss(fx_test, y_test) == expected


# Smoothed Hinge Loss
@pytest.mark.parametrize(
    "fx_test, y_test, expected",
    [(fx, y, 0.5 - fx * y) if fx * y <= 0 else
     (fx, y, 0) if fx * y >= 1 else
     (fx, y, (1 - y * fx) ** 2 / 2)
     for fx in rango_test
     for y in clases]
)
def test_smooth_hinge_loss(fx_test, y_test, expected):
    assert fc.smooth_hinge_loss(fx_test, y_test) == expected


# Modified Square Loss
@pytest.mark.parametrize(
    "fx_test, y_test, expected",
    [(fx, y, (1 - fx * y) ** 2) if 1 - y * fx >= 0 else
     (fx, y, 0)
     for fx in rango_test
     for y in clases]
)
def test_mod_square_loss(fx_test, y_test, expected):
    assert fc.mod_square_loss(fx_test, y_test) == expected


# Exponential Loss
@pytest.mark.parametrize(
    "fx_test, y_test, expected",
    [(fx, y, np.exp(-y * fx))
     for fx in rango_test
     for y in clases]
)
def test_exp_loss(fx_test, y_test, expected):
    assert fc.exp_loss(fx_test, y_test) == expected


# Log loss
@pytest.mark.parametrize(
    "fx_test, y_test, expected",
    [(fx, y, np.log(1 + np.exp(-y * fx)))
     for fx in rango_test
     for y in clases]
)
def test_log_loss(fx_test, y_test, expected):
    assert fc.log_loss(fx_test, y_test) == expected


"""
# Based on Sigmoid Loss
@pytest.mark.parametrize(
    "fx_test, y_test, gamma_test, theta_test, lambda_test, expected",
    [(fx, y, gamma, th, lambd, (1.2 - gamma) - gamma * y * fx) if -1 <= y * fx <= 0 else
     (fx, y, gamma, th, lambd, (gamma - lambd * y * fx) / (1 - th)) if th < y * fx <= 1 else
     (fx, y, gamma, th, lambd, (1.2 - gamma) - (1.2 - lambd) - (1.2 - 2 * gamma) * y * fx / th) if 0 <= y * fx <= th
     else (fx, y, gamma, th, lambd, None)
     for gamma in rango_test
     for lambd in rango_test
     for th in np.linspace(0.1, 1 - 1e-10, 5, dtype=float)
     for fx in rango_test
     for y in clases]
)
def test_sigmoid_loss(fx_test,
                      y_test,
                      gamma_test,
                      theta_test,
                      lambda_test,
                      expected):
    assert fc.sigmoid_loss(fx_test,
                           y_test,
                           gamma_test,
                           theta_test,
                           lambda_test) == expected
"""


# Log loss
@pytest.mark.parametrize(
    "fx_test, y_test, expected",
    [(fx, y, 1 - y * fx) if 0 <= y * fx <= 1 else
     (fx, y, 1 - np.sign(y * fx))
     for fx in rango_test
     for y in clases]
)
def test_phi_learning(fx_test, y_test, expected):
    assert fc.phi_learning(fx_test, y_test) == expected


# Ramp Loss
@pytest.mark.parametrize(
    "fx_test, y_test, s_test, c_test, expected",
    [(fx, y, s, c, (min(1 - s, max(1 - fx * y, 0)) + min(1 - s, max(1 + fx * y, 0)) + c))
     for s in np.linspace(-1 + 1e-10, 0, 25, dtype=float)
     for c in rango_test
     for fx in rango_test
     for y in clases]
)
def test_ramp_loss(fx_test, y_test, s_test, c_test, expected):
    assert fc.ramp_loss(fx_test, y_test, s_test, c_test) == expected


# Smooth non-convex loss
@pytest.mark.parametrize(
    "fx_test, y_test, lamb_test, expected",
    [(fx, y, lamb, 1 - np.tanh(lamb * y * fx))
     for lamb in np.linspace(-1 + 1e-10, 1, 25, dtype=float)
     for fx in rango_test
     for y in clases]
)
def test_smooth_non_convex_loss(fx_test, y_test, lamb_test, expected):
    assert fc.smooth_non_convex_loss(fx_test, y_test, lamb_test) - expected < 1e-10


# 2-layer Neural New-works
@pytest.mark.parametrize(
    "fx_test, y_test, expected",
    [(fx, y,  (1 - 1/(1+np.exp(-y * fx)))**2)
     for fx in rango_test
     for y in clases]
)
def test_layer_neural(fx_test, y_test, expected):
    assert fc.layer_neural(fx_test, y_test) == expected


# Logistic difference Loss
@pytest.mark.parametrize(
    "fx_test, y_test, mu_test, expected",
    [(fx, y, mu, ((np.log(1 + np.exp(-y * fx))) - (np.log(1 + np.exp(-y * fx - mu)))))
     for mu in rango_test
     for fx in rango_test
     for y in clases]
)
def test_logistic_difference_loss(fx_test, y_test, mu_test, expected):
    assert fc.logistic_difference_loss(fx_test, y_test, mu_test) == expected


# Smoothed 0-1 Loss
@pytest.mark.parametrize(
    "t_test, expected",
    [(t, 0) if t > 1 else
     (t, 1) if t < -1 else
     (t, (1/4)*t**3 - (3/4)*t + 0.5)
     for t in rango_test]
)
def test_smooth01(t_test, expected):
    assert fc.smooth01(t_test) == expected
