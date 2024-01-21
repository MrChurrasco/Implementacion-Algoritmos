import funciones_costo as fc
import pytest
import numpy as np

##
rango_test = np.linspace(-10, 10, 100, dtype=float)
precision = 1e-15


# Square Loss
@pytest.mark.parametrize(
    "t_test",
    rango_test
)
def test_square_loss_positive_always(t_test):
    assert fc.square_loss(t_test) > 0


@pytest.mark.parametrize(
    "t_test, expected",
    [(t, (1 - t) ** 2)
     for t in rango_test]
)
def test_square_loss(t_test, expected):
    assert fc.square_loss(t_test) - expected < precision


# Hinge Loss
@pytest.mark.parametrize(
    "t_test, expected",
    [(t, 1 - t) if t < 1
     else (t, 0.)
     for t in rango_test]
)
def test_hinge_loss(t_test, expected):
    assert fc.hinge_loss(t_test) - expected < precision


# Smoothed Hinge Loss
@pytest.mark.parametrize(
    "t_test, expected",
    [(t, 0.5 - t) if t <= 0 else
     (t, 0) if t >= 1 else
     (t, (1 - t) ** 2 / 2)
     for t in rango_test]
)
def test_smooth_hinge_loss(t_test, expected):
    assert fc.smooth_hinge_loss(t_test) - expected < precision


# Modified Square Loss
@pytest.mark.parametrize(
    "t_test, expected",
    [(t, (1 - t) ** 2) if 1 - t >= 0 else
     (t, 0)
     for t in rango_test]
)
def test_mod_square_loss(t_test, expected):
    assert fc.mod_square_loss(t_test) - expected < precision


# Exponential Loss
@pytest.mark.parametrize(
    "t_test, expected",
    [(t, np.exp(-t))
     for t in rango_test]
)
def test_exp_loss(t_test, expected):
    assert fc.exp_loss(t_test) - expected < precision


# Log loss
@pytest.mark.parametrize(
    "t_test, expected",
    [(t, np.log(1 + np.exp(-t)))
     for t in rango_test]
)
def test_log_loss(t_test, expected):
    assert fc.log_loss(t_test) - expected < precision


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
    "t_test, expected",
    [(t, 1 - t) if 0 <= t <= 1 else
     (t, 1 - np.sign(t))
     for t in rango_test]
)
def test_phi_learning(t_test, expected):
    assert fc.phi_learning(t_test) - expected < precision


# Ramp Loss
@pytest.mark.parametrize(
    "t_test, s_test, c_test, expected",
    [(t, s, c, (min(1 - s, max(1 - t, 0)) + min(1 - s, max(1 + t, 0)) + c))
     for s in np.linspace(-1 + 1e-10, 0, 25, dtype=float)
     for c in rango_test
     for t in rango_test]
)
def test_ramp_loss(t_test, s_test, c_test, expected):
    assert fc.ramp_loss(t_test, s_test, c_test) - expected < precision


# Smooth non-convex loss
@pytest.mark.parametrize(
    "t_test, lamb_test, expected",
    [(t, lamb, 1 - np.tanh(lamb * t))
     for lamb in np.linspace(-1 + 1e-10, 1, 25, dtype=float)
     for t in rango_test]
)
def test_smooth_non_convex_loss(t_test, lamb_test, expected):
    assert fc.smooth_non_convex_loss(t_test, lamb_test) - expected < precision


# 2-layer Neural New-works
@pytest.mark.parametrize(
    "t_test, expected",
    [(t,  (1 - 1/(1+np.exp(-t)))**2)
     for t in rango_test]
)
def test_layer_neural(t_test, expected):
    assert fc.layer_neural(t_test) - expected < precision


# Logistic difference Loss
@pytest.mark.parametrize(
    "t_test, mu_test, expected",
    [(t, mu, ((np.log(1 + np.exp(-t))) - (np.log(1 + np.exp(-t - mu)))))
     for mu in rango_test
     for t in rango_test]
)
def test_logistic_difference_loss(t_test, mu_test, expected):
    assert fc.logistic_difference_loss(t_test, mu_test) - expected < precision


# Smoothed 0-1 Loss
@pytest.mark.parametrize(
    "t_test, expected",
    [(t, 0) if t > 1 else
     (t, 1) if t < -1 else
     (t, (1/4)*t**3 - (3/4)*t + 0.5)
     for t in rango_test]
)
def test_smooth01(t_test, expected):
    assert fc.smooth01(t_test) - expected < precision
