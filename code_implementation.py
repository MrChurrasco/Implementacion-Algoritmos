import numpy as np
from numpy.random import rand, seed
from numpy.linalg import norm
from matplotlib import pyplot as plt
from time import time
from keras.datasets import mnist
import pandas as pd


# Funciones dentro de funciones
def v(arr):
    thresholds = (-1, 1)
    funcs = (lambda t: 1, lambda t: 1 / 4 * t ** 3 - 3 / 4 * t + 1 / 2, lambda t: 0)
    masks = np.array([arr < threshold for threshold in thresholds])
    masks = [masks[0]] + [x for x in masks[1:] & ~masks[:-1]] + [~masks[-1]]
    result = np.empty_like(arr)
    for mask, func in zip(masks, funcs):
        result[mask] = func(arr[mask])
    return result


def dv(arr):
    thresholds = (-1, 1)
    funcs = (lambda t: 0, lambda t: 3 / 4 * t ** 2 - 3 / 4, lambda t: 0)
    # funcs=(lambda t: 0, lambda t: -1/4*np.pi*np.cos(np.pi/2*t), lambda t: 0)
    masks = np.array([arr < threshold for threshold in thresholds])
    masks = [masks[0]] + [x for x in masks[1:] & ~masks[:-1]] + [~masks[-1]]
    result = np.empty_like(arr)
    for mask, func in zip(masks, funcs):
        result[mask] = func(arr[mask])
    return result


def backtracking(f, g, xk, pk, alpha=1, beta=0.5, c=1e-4):
    rhs = c * g(xk) @ pk
    fxk = f(xk)
    while f(xk + alpha * pk) > fxk + alpha * rhs:
        alpha = beta * alpha
    return alpha


def f(x, c, m, lam, cx=None):
    if cx is None:
        cx = c @ x
    return 1 / m * v(cx).sum() + lam / 2 * norm(x) ** 2


def df(x, c, m, lam, cx=None):
    if cx is None:
        cx = c @ x
    return 1 / m * c.T @ dv(cx) + lam * x


def s2_v2(t, s):
    # Ensure t and s are NumPy arrays of the same shape
    t = np.array(t)
    s = np.array(s)

    # Check conditions element-wise and compute the result element-wise
    condition = (-1 <= t) & (t <= 1) & (s != 0)
    result = np.zeros_like(t)
    result[condition] = (3 / 2) * t[condition] * s[condition]
    return result


def grad(x0, fun, dfun, tol=1e-6):
    t = time()
    xk = x0.copy()
    continua = True
    k = 0
    while continua:
        pk = -dfun(xk)
        npk = norm(pk)
        if npk < tol:
            continua = False
        paso = backtracking(fun, dfun, xk, pk)
        xk = xk + paso * pk
        k += 1
        # print(k,paso)
        if paso < 1e-14:
            print("here")
    return xk, k, time() - t, fun(xk), norm(dfun(xk))


def s2_v(t, s):
    # Ensure t and s are NumPy arrays of the same shape
    # t = np.array(t)
    # s = np.array(s)

    # Check conditions element-wise and compute the result element-wise
    condition = (-1 <= t) & (t <= 1) & (s != 0)
    result = np.where(condition, (3 / 2) * t * s, 0)
    # result = np.where(condition, 1/8*np.pi**2*np.sin(np.pi/2*t) * s, 0)
    return result


def alg1(x0, sigma, t0, c, m, n, lam, tol=1e-6, beta=0.5, gam=2):
    t = time()
    xk = x0.copy()
    k = 0
    reduced = 0
    continua = True
    tauk = t0
    # CCT=C@C.T#np.expand_dims(c,1)@np.expand_dims(c,0)
    nc2 = np.linalg.norm(c.T, axis=0) ** 2
    while continua:
        Cxk = c @ xk
        wk = df(xk, c, m, Cxk)
        if norm(wk) < tol:
            continua = False
        s = s2_v(np.expand_dims(Cxk, 1), np.ones((m, 1)))
        s0 = s.flatten() * nc2.flatten()
        rhok = -np.sum(s0[s0 < 0])
        # print(k,rhok)
        dk = np.linalg.solve(c.T @ (s * c) / m + (lam + rhok) * np.eye(n), -wk)
        if reduced >= 2:
            tauk = gam * tauk
        else:
            reduced += 1
        rhs = sigma * dk @ wk
        tauk = max(tauk, 1e-6)
        fxk = f(xk, c, m, Cxk)
        while f(xk + tauk * dk, c, m, lam) > fxk + tauk * rhs:
            tauk = tauk * beta
            reduced = 0
        xk = xk + tauk * dk
        k += 1
        # print(k,tauk,f(xk+tauk*dk),norm(wk))
    return xk, k, time() - t, f(xk, c, m, lam), norm(df(xk, c, m, lam))


## Funciones Main
def dibuja(n0, n1, sol, fun, testX, testy, noise=0.001):
    select_test = np.where((testy == n0) + (testy == n1))[0]
    x_test = testX[select_test].reshape(select_test.shape[0], 28 ** 2)  # /255.
    x_test = (x_test - x_test.mean().astype(np.float32)) / x_test.std().astype(np.float32)
    x_test += np.random.randn(*x_test.shape) * noise
    x_test = np.concatenate([np.ones([x_test.shape[0], 1]), x_test], axis=1)
    y_test = testy[select_test] == n1
    y_test0 = (y_test == 0)

    ind = np.where((fun(x_test @ sol) >= 0.5) != y_test0)[0]
    num_row, num_col = 2, int(np.ceil(ind.shape[0] / 2))
    fig, axes = plt.subplots(num_row, num_col, figsize=(1.5 * num_col / 2, 2 * num_row / 2))
    for i in range(ind.shape[0]):
        ax = axes[i // num_col, i % num_col]
        ax.imshow(x_test[ind[i]][1:].reshape(28, 28), cmap='gray_r')
        ax.axis(False)
        ax.set_title('Label: {} '.format(testy[select_test][ind[i]]), fontsize=10)
    plt.tight_layout()
    # plt.savefig('Misclassified_'+str(n0)+'_'+str(n1)+'_'+str(tam)+'_'+str(noise)+'.pdf',bbox_inches='tight')
    plt.show()


def train(n0, n1, v_fun, dv_fun, fun, dfun, backtracking_fun,
          trainy, trainx, testy, testx,
          tam=None, noise=0.001, lam=0.001, toler=1e-3, beta=0.2,
          gam=2, gradient=False, dibuja_ind=False):

    # Tamao dataset
    if tam is None:
        select = np.where((trainy == n0) + (trainy == n1))[0]
        tam = select.shape[0]
    else:
        select = np.where((trainy == n0) + (trainy == n1))[0][:tam]
    x_train = trainx[select].reshape(tam, 28 ** 2)  # /255.
    x_train = (x_train - x_train.mean().astype(np.float32)) / x_train.std().astype(
        np.float32)  # Image preprocessing: we standarized by substracting the mean and dividing by the standard deviation
    select_test = np.where((testy == n0) + (testy == n1))[0]
    x_test = testx[select_test].reshape(select_test.shape[0], 28 ** 2)  # /255.
    x_test = (x_test - x_test.mean().astype(np.float32)) / x_test.std().astype(np.float32)

    if noise is not None:
        ruido = np.random.randn(*x_train.shape) * noise
        smallest_sv = np.linalg.svd(x_train + ruido)[1].min()
        if smallest_sv < 1e-6:
            ruido = np.random.randn(*x_train.shape) * noise
            smallest_sv = np.linalg.svd(x_train + ruido)[1].min()
        x_train += ruido
        x_test += np.random.randn(*x_test.shape) * noise
        print('smallest singular value =', smallest_sv)
    else:
        smallest_sv = np.linalg.svd(x_train)[1].min()
    x_train = np.concatenate([np.ones([x_train.shape[0], 1]), x_train], axis=1)
    y_train = trainy[select] == n1
    x_test = np.concatenate([np.ones([x_test.shape[0], 1]), x_test], axis=1)
    y_test = testy[select_test] == n1

    m = y_train.shape[0]
    print('Numbers:', n0, 'and', n1)
    print('m =', m)
    print('noise =', noise)
    print('lambda = ', lam)

    y = 2 * y_train - 1
    C = np.expand_dims(y, 1) * x_train.copy()
    n = x_train.shape[1]

    # cTx=(-4/3*lam+(16/9*lam**2+4*nc**4)**.5)/(2*nc**2)
    # sol=3/4/lam*c*(1-cTx**2)

    # def S2V(t, s):
    #     if t < -1 or t > 1 or s == 0:
    #         return 0
    #     else:
    #         return (3/2) * t * s

    x0 = 2 * (rand(n) - .5)

    x0 = 2 * (rand(n) - .5)
    sol_alg1 = alg1(x0, sigma=0.2, t0=1, c=C, m=m, n=n, lam=lam, beta=beta, gam=gam, tol=toler)
    print('Alg1 done! [n,m] = ', [n, m])
    print('f(x0)= ', fun(x0, C, m))

    print('Alg1 :', sol_alg1[1:])  # ,norm(sol-sol_alg1[0]))

    y_train0 = (y_train == 0)
    y_test0 = (y_test == 0)

    aciertos_train = ((v_fun(x_train @ sol_alg1[0]) >= 0.5) == y_train0).sum()
    aciertos_test = ((v_fun(x_test @ sol_alg1[0]) >= 0.5) == y_test0).sum()

    print(
        f'Aciertos en el conjunto de entrenamiento: {aciertos_train} de {y_train.shape[0]} ({(aciertos_train / y_train.shape[0]):2.2%})')
    print(
        f'Aciertos en el conjunto de test: {aciertos_test} de {y_test.shape[0]} ({(aciertos_test / y_test.shape[0]):2.2%})')

    if dibuja_ind:
        ind = np.where((v_fun(x_test @ sol_alg1[0]) >= 0.5) != y_test0)[0]
        num_row, num_col = 2, int(np.ceil(ind.shape[0] / 2))
        fig, axes = plt.subplots(num_row, num_col, figsize=(1.5 * num_col / 2, 2 * num_row / 2))
        for i in range(ind.shape[0]):
            ax = axes[i // num_col, i % num_col]
            ax.imshow(x_test[ind[i]][1:].reshape(28, 28), cmap='gray_r')
            ax.axis(False)
            ax.set_title('Label: {} '.format(testy[select_test][ind[i]]), fontsize=10)
        plt.tight_layout()
        plt.savefig('Misclassified_' + str(n0) + '_' + str(n1) + '_' + str(tam) + '_' + str(noise) + '.pdf',
                    bbox_inches='tight')
        plt.show()

    if gradient:
        sol_grad = grad(x0, tol=toler)
        print('Grad done! [n,m] = ', [n, m])
        print('Grad :', sol_grad[1:])  # ,norm(sol-sol_grad[0]))

        aciertos_grad_train = ((v_fun(x_train @ sol_grad[0]) >= 0.5) == y_train0).sum()
        aciertos_grad_test = ((v_fun(x_test @ sol_grad[0]) >= 0.5) == y_test0).sum()
        print(
            f'Aciertos en el conjunto de entrenamiento: {aciertos_grad_train} de {y_train.shape[0]} ({(aciertos_grad_train / y_train.shape[0]):2.2%})')
        print(
            f'Aciertos en el conjunto de test: {aciertos_grad_test} de {y_test.shape[0]} ({(aciertos_grad_test / y_test.shape[0]):2.2%})')
        return smallest_sv, sol_alg1, y_train.shape[0], aciertos_train, y_test.shape[
            0], aciertos_test, sol_grad, aciertos_grad_train, aciertos_grad_test
    else:
        return smallest_sv, sol_alg1, y_train.shape[0], aciertos_train, y_test.shape[0], aciertos_test


def main():
    pd.set_option('display.max_columns', None)
    pd.set_option('display.max_rows', None)
    pd.set_option('display.width', 300)

    seed(1)

    lam = 0.001
    toler = 1e-3
    # lam=0.005
    # m=6000

    # C=2*(rand(m,n)-.5)

    (trainX, trainy), (testX, testy) = mnist.load_data()

    tam = 5000

    n0 = 3
    n1 = 8

    noise = 1e-3

    ejecuta_noise = False
    if ejecuta_noise:
        tolerancia = 1e-3
        numeros = [1, 7]
        reps = 10
        ruido = [.5, 1e-1, 1e-2, 1e-3, None]
        tams = [None]
        R = list()
        n0 = numeros[0]
        n1 = numeros[1]
        for tam in tams:
            for noise in ruido:
                for r in range(reps):
                    print('repet', r + 1, 'of', reps)
                    seed(r)
                    R.append([r, tam, noise,
                              train(n0, n1, v, dv, f, df, tam=tam, noise=noise, gradient=True, toler=tolerancia)])
                    np.savez('Results_MNIST_noise_20231011', R)
    else:
        with np.load('Results_MNIST_noise_20231011.npz', allow_pickle=True) as datos:
            R = datos['arr_0']
        S = ([
            [R[k][3][2], R[k][2], R[k][3][2] - R[k][3][3], R[k][3][2] - R[k][3][7], R[k][3][4], R[k][3][4] - R[k][3][5],
             R[k][3][4] - R[k][3][8], round(R[k][3][5] / R[k][3][4] * 100, 2), round(100 * R[k][3][8] / R[k][3][4], 2),
             round(R[k][3][1][2], 2), round(R[k][3][6][2], 2), R[k][3][1][1], R[k][3][6][1]] for k in range(len(R))])
        pdS = pd.DataFrame(S)
        pdS.columns = ['m', 'noise', 'f tr A1', 'f tr GD', 'test', 'f te A1', 'f te GD', '%succ Alg1', '%succ GD',
                       'time Alg1', 'time GD', 'it Alg1', 'it GD']
        print(pdS)
        print()

    if 0:
        ejecuta_noise2 = True
        if ejecuta_noise2:
            tolerancia = 1e-3
            numeros = [[3, 8], [8, 9], [1, 2], [5, 6], [6, 8], [4, 9]]
            reps = 10
            ruido = [1e-2, 1e-3, None]
            tams = [None]
            R = list()
            n0 = numeros[0]
            n1 = numeros[1]
            for tam in tams:
                for noise in ruido:
                    for r in range(reps):
                        print('repet', r + 1, 'of', reps)
                        seed(r)
                        R.append([r, tam, noise, train(n0, n1, tam=tam, noise=noise, gradient=True, toler=tolerancia)])
                        np.savez('Results_MNIST_noise_20231011', R)
        else:
            with np.load('Results_MNIST_noise_20231011.npz', allow_pickle=True) as datos:
                R = datos['arr_0']
            S = ([[R[k][3][2], R[k][2], R[k][3][2] - R[k][3][3], R[k][3][2] - R[k][3][7], R[k][3][4],
                   R[k][3][4] - R[k][3][5], R[k][3][4] - R[k][3][8], round(R[k][3][5] / R[k][3][4] * 100, 2),
                   round(100 * R[k][3][8] / R[k][3][4], 2), round(R[k][3][1][2], 2), round(R[k][3][6][2], 2),
                   R[k][3][1][1], R[k][3][6][1]] for k in range(len(R))])
            pdS = pd.DataFrame(S)
            pdS.columns = ['m', 'noise', 'f tr A1', 'f tr GD', 'test', 'f te A1', 'f te GD', '%succ Alg1', '%succ GD',
                           'time Alg1', 'time GD', 'it Alg1', 'it GD']
            print(pdS)
            print()

        if 1:
            ejecuta = True
            if ejecuta:
                tolerancia = 1e-3
                numeros = [[1, 7], [3, 8], [8, 9], [1, 2], [5, 6], [6, 8], [4, 9]]
                ruido = [1e-2, 1e-3, None]
                tams = [100, 500, 1000, 5000, None]
                R = list()
                for num in numeros:
                    n0 = num[0]
                    n1 = num[1]
                    for tam in tams:
                        for noise in ruido:
                            R.append(
                                [num, tam, noise, train(n0, n1, tam=tam, noise=noise, gradient=True, toler=tolerancia)])
                            np.savez('Results_MNIST_1', R)
            else:
                with np.load('Results_MNIST_1.npz', allow_pickle=True) as datos:
                    R = datos['arr_0']
                S = ([[R[k][0][0], R[k][0][1], R[k][3][2], R[k][2], R[k][3][2] - R[k][3][3], R[k][3][2] - R[k][3][7],
                       R[k][3][4], R[k][3][4] - R[k][3][5], R[k][3][4] - R[k][3][8],
                       round(R[k][3][5] / R[k][3][4] * 100, 2), round(100 * R[k][3][8] / R[k][3][4], 2),
                       round(R[k][3][1][2], 2), round(R[k][3][6][2], 2), R[k][3][1][1], R[k][3][6][1]] for k in
                      range(len(R))])
                pdS = pd.DataFrame(S)
                pdS.columns = ['n0', 'n1', 'm', 'noise', 'f tr A1', 'f tr GD', 'test', 'f te A1', 'f te GD',
                               '%succ Alg1', '%succ GD', 'time Alg1', 'time GD', 'it Alg1', 'it GD']
                print(pdS)
                print()
                print(pdS.loc[np.where(pdS.loc[:, 'm'] > 5000)])

        if 1:
            ejecuta = True
            if ejecuta:
                tolerancia = 1e-3
                numeros = [[3, 8], [8, 9], [1, 2], [5, 6], [6, 8], [4, 9]]
                ruido = [1e-2, 1e-3, None]
                tams = [None]
                R = list()
                for num in numeros:
                    n0 = num[0]
                    n1 = num[1]
                    for tam in tams:
                        for noise in ruido:
                            R.append(
                                [num, tam, noise, train(n0, n1, tam=tam, noise=noise, gradient=True, toler=tolerancia)])
                            np.savez('Results_MNIST_0', R)
            else:
                with np.load('Results_MNIST_0.npz', allow_pickle=True) as datos:
                    R = datos['arr_0']
                S = ([[R[k][0][0], R[k][0][1], R[k][3][2], R[k][2], R[k][3][2] - R[k][3][3], R[k][3][2] - R[k][3][7],
                       R[k][3][4], R[k][3][4] - R[k][3][5], R[k][3][4] - R[k][3][8],
                       round(R[k][3][5] / R[k][3][4] * 100, 2), round(100 * R[k][3][8] / R[k][3][4], 2),
                       round(R[k][3][1][2], 2), round(R[k][3][6][2], 2), R[k][3][1][1], R[k][3][6][1]] for k in
                      range(len(R))])
                pdS = pd.DataFrame(S)
                pdS.columns = ['n0', 'n1', 'm', 'noise', 'f tr A1', 'f tr GD', 'test', 'f te A1', 'f te GD',
                               '%succ Alg1', '%succ GD', 'time Alg1', 'time GD', 'it Alg1', 'it GD']
                print(pdS)
                print()
                print(pdS.loc[np.where(pdS.loc[:, 'm'] > 5000)])
