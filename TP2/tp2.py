import numpy as np
import matplotlib.pyplot as plt
import numpy.linalg as lg


def fs(Y, A):
    res = lg.inv(A.T @ A) @ A.T @ Y
    return res


def residus(Y, A, X):
    return Y - A @ X


def dataset(file: str):
    res = np.loadtxt(file)
    return res


def choix():
    L = dataset('data/choix_regression.txt')
    x = L[:, 0:1]
    y = L[:, 1:2]

    M = np.ones(x.shape)
    # A = np.array([x**3, x**2, x, M])
    A = np.array([x ** 3, x ** 2, x, M])
    A = np.hstack(A)

    X = fs(y, A)
    print(X)
    plt.plot(x, X[0, 0] * x ** 3 + X[1, 0] * x ** 2 + X[2, 0] * x + X[3, 0])
    plt.show()

    V = residus(y, A, X)
    print(np.mean(V), np.std(V))


def aberrant():
    L = dataset('data/points_aberrants.txt')
    x = L[:, 0:1]
    y = L[:, 1:2]

    M = np.ones(x.shape)
    A = np.hstack(np.array([x, M]))

    X = fs(y, A)
    plt.scatter(x, y, color='red')
    plt.plot(x, X[0, 0] * x + X[1, 0])
    plt.title('Avec un point aberrant')
    plt.show()

    V = residus(y, A, X)
    print(np.mean(V), np.std(V))

    print(np.argmax(V))

    x = x[1:]
    y = y[1:]

    Ab = np.hstack(np.array([x, M[1:]]))
    Xb = fs(y, Ab)
    plt.scatter(x, y, color='red')
    plt.plot(x, Xb[0, 0] * x + Xb[1, 0])
    plt.title('Sans le point aberrant')
    plt.show()

    Vb = residus(y, Ab, Xb)
    print(np.mean(Vb), np.std(Vb))


def temperature():
    L = dataset('data/temperature.txt')
    x = L[:, 0:1]
    y = L[:, 1:2]

    M = np.ones(x.shape)
    A = np.hstack(np.array([x, M]))
    X = fs(y, A)

    gamma = X[0, 0]
    T0 = X[1, 0]

    plt.scatter(x, y, color='red')
    plt.plot(x, gamma * x + T0, label='$T(z) = \Gamma z + T_0$')
    plt.title('$\Gamma = $' + str(gamma) + "$ ; T_0 = $" + str(T0))
    plt.legend()
    plt.show()


def chute():
    L = dataset('data/chute_bille.txt')
    x = L[:, 0:1]
    y = L[:, 1:2]

    M = np.ones(x.shape)
    A = np.hstack(np.array([x ** 2, x, M]))
    X = fs(y, A)

    plt.scatter(x, y, color='red')
    plt.plot(x, X[0, 0] * x ** 2 + X[1, 0] * x + X[2, 0])
    plt.show()


def param():
    L = dataset('data/courbe_parametree.txt')
    t = L[:, 0:1]
    x = L[:, 1:2]
    y = L[:, 2:3]

    Ax = pow(np.sin(t), 3)
    Xx = fs(x, Ax)

    # plt.plot(x, y, '.', color='red')
    # plt.plot(t, Xx[0, 0] * Ax)
    # plt.show()

    M = np.ones(y.shape)
    C = np.cos(t)
    Ay = np.hstack(np.array([C ** 4, C ** 3, C ** 2, C, M]))
    Xy = fs(y, Ay)

    # plt.scatter(t, y, color='red')
    # plt.plot(t, Xy[0, 0] * C ** 4 + Xy[1, 0] * C ** 3
    #          + Xy[2, 0] * C ** 2 + Xy[3, 0] * C + Xy[4, 0])
    # plt.show()

    plt.plot(x, y, '.r')
    plt.plot(Xx[0, 0] * Ax, Xy[0, 0] * C ** 4 + Xy[1, 0] * C ** 3
             + Xy[2, 0] * C ** 2 + Xy[3, 0] * C + Xy[4, 0], '.b')
    plt.show()


def maree():
    L = dataset('data/maree_brest.txt')
    t = L[:, 0:1]
    y = L[:, 1:2]

    M = np.ones(t.shape)

    periods = 3600 * np.array([12.42, 12.66, 12.00, 11.97, 25.82,
                               23.93, 24.07, 26.87])
    omegas = np.array([2 * np.pi / p for p in periods])

    M = np.ones(t.shape)
    A = np.array(t.shape)

    C = np.array([np.cos(o*t) for o in omegas])
    S = np.array([np.sin(o * t) for o in omegas])

    A = np.hstack([M])
    for i in range(8):
        A = np.hstack([A, C[i]])
    for i in range(8):
        A = np.hstack([A, S[i]])

    X = fs(y, A)[:, 0]

    h0 = X[0]
    Bs = X[1:9]
    Cs = X[9:]

    Amps = pow(pow(Bs, 2) + pow(Cs, 2), .5)
    phis = np.arctan(-Cs/Bs)

    est = []
    for i in range(8):
        tmp = Amps[i] * np.cos(omegas[i] * t + phis[i])
        est.append(tmp)

    plt.plot(t, y, '.r')
    plt.plot(t, h0 + sum(est), '.b')
    plt.show()


if __name__ == '__main__':
    # choix()
    # aberrant()
    # temperature()
    # chute()
    # param()
    maree()
