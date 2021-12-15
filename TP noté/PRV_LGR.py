import numpy as np
import numpy.linalg as lg
import matplotlib.pyplot as plt


def readData(file: str):
    """
    Lecture des données spatiales depuis un fichier
    :param file: chemin vers le fichier de données
    :type file: String
    :return: données spatiales (x,y,z)
    """
    data = np.load(file)
    x = data[:, 0:1]
    y = data[:, 1:2]
    z = data[:, 2:3]
    return x, y, z


def statsSerie(S):
    """
    Donne les statistiques d'une série S
    :param S: jeu de données sur lequel on établit les stats
    :return: moyenne, écart-type, minimum et maximum
    """
    return np.mean(S), np.std(S), np.min(S), np.max(S)


def showCloud(x, y, z, title=""):
    """
    Affichage du nuage de points (x,y,z)
    :param x: vecteur des abscisses
    :param y: vecteur des ordonnées
    :param z: vecteur des cotes
    :return:
    """
    fig = plt.figure()
    ax = fig.gca(projection='3d')
    ax.scatter(x, y, z)
    plt.title(title)
    plt.xlabel("x")
    plt.ylabel("y")
    plt.show()


def plotRes(v, residus, title=''):
    plt.plot(v, residus, 'b.')
    plt.xlabel('x')
    plt.ylabel('résidus')
    plt.title(title)
    plt.show()


def leastSquares(Y, A):
    """
    Méthode des moindres carrés pour un système AX = Y
    :param Y: matrice des résultats
    :param A: matrice des valeurs
    :return: matrice X
    """
    res = lg.inv(A.T @ A) @ A.T @ Y
    return np.asarray(res)


def equaCyl(y, z, residus=False):
    """
    Calcule l'équation d'un cylindre (y-y0)^2 + (z-z0)^2 = r0^2 réduit à l'équation AX = Y
    :param y: vecteur des ordonnées
    :param z: vecteur des cotes
    :param residus: calcul des résidus (Y/N)
    :return: coefficients (x0, y0, r0)
    """
    vy = y.reshape(-1, 1)
    vz = z.reshape(-1, 1)
    A = np.hstack((2 * vy, 2 * vz, np.ones(vy.shape)))
    Y = np.power(vy, 2) + np.power(vz, 2)
    tmp = leastSquares(Y, A)
    res = np.array([tmp[0][0], tmp[1, 0], pow(tmp[2][0] + tmp[0][0] ** 2 + tmp[1][0] ** 2, .5)])
    if residus:
        return res, Y - A @ tmp
    else:
        return res


def drawCompare(x, y, z, y0, z0, r0):
    """
    Affiche le cylindre et son modèle numérique afin de les comparer graphiquement
    :param x: abscisses du cylindre
    :param y: ordonnées du cylindre
    :param z: cotes du cylindre
    :param y0: origine en y du modèle
    :param z0: origine en z du modèle
    :param r0: rayon du modèle
    """
    n = 400
    U, V = np.meshgrid(np.linspace(-.5 * np.pi, .5 * np.pi, n), np.linspace(np.min(x), np.max(x), n))
    X = V
    Y = r0 * np.sin(U) + y0
    Z = r0 * np.cos(U) + z0
    fig = plt.figure()
    ax = fig.gca(projection='3d')
    ax.scatter(x, y, z)
    ax.plot_wireframe(X, Y, Z, color='gray')
    plt.title("Comparaison de la nef et son modèle")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.show()


def statsMoustache(data, title=""):
    """
    Affiche les profondeurs sous forme de boîtes à moustaches
    :param data: liste des profondeurs mesurées
    :param title: titre de la figure
    """
    plt.boxplot(data, showmeans=True)
    plt.title(title)
    plt.show()


def simpleCylindre():
    ## code utilisé dans la première partie (cylindre simple)
    x, y, z = readData('data/nef.npy')
    # showCloud(x, y, z, title="Points de la nef")
    R, residu = equaCyl(y, z, residus=True)
    y0 = R[0]
    z0 = R[1]
    r0 = R[2]
    # plotRes(x, residu, title="Résidus des moindres carrés")
    # drawCompare(x, y, z, y0, z0, r0)
    statsMoustache(residu,"Détermination des statistiques des résidus")


def decoupage2(x, y, z, n):
    """
    Découpe le demi-cylindre en n portions d'égale longueur en x
    :param x: abscisses du cylindre
    :param y: ordonnées du cylindre
    :param z: cotes du cylindre
    :param n: nombre de portions
    :return: Les variables nommées dynamiquement "Intervallei" pour 0<= i < n
    """
    xmin = np.min(x)
    xmax = np.max(x)
    bornes = np.linspace(xmin, xmax, n)
    # print(xmin, xmax, bornes)
    for j in range(len(bornes) - 1):
        globals()["Intervalle" + str(j)] = []
    for i in range(len(x)):
        for j in range(len(bornes) - 1):
            if (x[i] >= bornes[j]) & (x[i] < bornes[j + 1]):
                (globals()["Intervalle" + str(j)]).append(np.array([float(x[i]), float(y[i]), float(z[i])]))


if __name__ == '__main__':
    n = 31
    x, y, z = readData('data/nef.npy')
    S = equaCyl(y, z)
    y0 = S[0]
    z0 = S[1]
    r0 = S[2]

    decoupage2(x, y, z, n)
    R = []
    for i in range(n-1):
        I = globals()["Intervalle"+str(i)]
        I = np.asarray(I)
        # print(I)
        R.append(equaCyl(I[:, 1], I[:, 2]))

    R = np.asarray(R)
    # print(statsSerie(R[:, 0]-y0))
    # print(statsSerie(R[:, 1]-z0))
    # print(statsSerie(R[:, 2]-r0))
    # statsMoustache(R[:, 0]-y0, title="Écarts entre les ${y_0}^k$ et $y_0$")
    # statsMoustache(R[:, 1]-z0, title="Écarts entre les ${z_0}^k$ et $z_0$")
    # statsMoustache(R[:, 2]-r0, title="Écarts entre les ${r_0}^k$ et $r_0$")
    # plt.plot(R[:, 2], '--b.')
    fig, ax1 = plt.subplots()

    ax1.set_xlabel('x')
    ax1.set_ylabel('y')
    ax1.plot(R[:, 0], '--r.', label="y")
    plt.legend()
    ax1.tick_params(axis='y')

    ax2 = ax1.twinx()  # même axe x

    ax2.set_ylabel('z')
    ax2.plot(R[:, 1], '--g.', label="z")
    ax2.tick_params(axis='y')
    plt.title("Évolution des centres le long de la nef")
    plt.legend()
    plt.show()

