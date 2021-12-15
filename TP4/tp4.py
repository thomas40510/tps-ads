import numpy as np
import matplotlib.pyplot as plt
import numpy.linalg as lg


def dataset(file: str, isnpz=True):
    """
    Lecture d'un fichier de données
    :param file: fichier de données
    :param isnpz: le fichier est au format npz (Y/N)
    :return: contenu du fichier sous forme d'array np
    """
    if isnpz:
        res = np.load(file)
        return res
    else:
        res = np.loadtxt(file)
        return res


def readData(file: str, isnpz=True):
    """
    Lecture des données spatiales depuis un fichier
    :param file: fichier de données
    :param isnpz: le fichier est au format npz (Y/N)
    :return: données spatiales (x,y,z)
    """
    F = dataset(file, isnpz)
    if isnpz:
        x = F['x']
        y = F['y']
        z = F['z']
        return x, y, z
    else:
        x = np.asarray(F[:, 0:1])
        y = np.asarray(F[:, 1:2])
        z = np.asarray(F[:, 2:3])
        return x, y, z


def showPlan(file: str):
    """
    Affichage des données 3D contenues dans un fichier
    :param file: fichier contenant les données
    :return:
    """
    x, y, z = readData(file, file.__contains__('.npz'))

    X, Y = np.meshgrid(x, y)

    plt.pcolormesh(X, Y, z, cmap='gist_earth', shading='auto')
    plt.colorbar()
    plt.contour(X, Y, z, levels=20, colors='black')
    plt.title("Cartographie de la bathymétrie")
    plt.show()


def showCloud(x, y, z):
    """
    Affichage du nuage de points (x,y,z)
    :param x: vecteur des abscisses
    :param y: vecteur des ordonnées
    :param z: vecteur des cotes
    :return:
    """
    fig = plt.figure()
    ax = fig.gca(projection='3d')
    ax.scatter(x, y, c=z)
    plt.show()


def statsSerie(S):
    """
    Donne les statistiques d'une série z
    :param S: jeu de données sur lequel on établit les stats
    :return: moyenne, écart-type, minimum et maximum
    """
    return np.mean(S), np.std(S), np.min(S), np.max(S)


def leastSquares(Y, A):
    """
    Méthode des moindres carrés pour un système AX = Y
    :param Y: matrice des résultats
    :param A: matrice des valeurs
    :return: matrice X
    """
    res = lg.inv(A.T @ A) @ A.T @ Y
    return np.asarray(res)


def equaPlan(x, y, z, residus=False):
    """
    Calcule l'équation d'un plan ax + by + c = z réduit à l'équation AX = B
    :param x: vecteur des abscisses
    :param y: vecteur des ordonnées
    :param z: vecteur des cotes
    :param residus: calcul des résidus (Y/N)
    :return: coefficients (a,b,c)
    """
    vx = x.reshape(-1, 1)
    vy = y.reshape(-1, 1)
    A = np.hstack((vx, vy, np.ones(vx.shape)))
    B = z.reshape(-1, 1)
    res = leastSquares(B, A)
    if residus:
        return res, B - A @ res
    else:
        return res


def equaSphere(x, y, z):
    """
    Donne le centre et le rayon d'une sphère
    :param x: abscisses
    :param y: ordonnées
    :param z: cotes
    :return: centre (x0,y0,z0) et rayon r0
    """
    vx = x.reshape(-1, 1)
    vy = y.reshape(-1, 1)
    vz = z.reshape(-1, 1)
    A = np.hstack((2 * vx, 2 * vy, 2 * vz, np.ones(vx.shape)))
    B = x * x + y * y + z * z
    res = leastSquares(B, A)
    r0 = pow(res[0]**2 + res[1]**2 + res[2]**2 + res[3], .5)
    return res[:3], r0


if __name__ == '__main__':
    # showPlan('data/mnt.npz')
    # x, y, z = readData('data/mnt.npz')
    #
    # print(statsSerie(z))
    # print(equaPlan(X, Y, z, residus=False))
    # E = equaPlan(X, Y, z, residus=True)
    # print(statsSerie(E[1]))

    x, y, z = readData('data/sphere.txt', False)
    # print(x, y, z)
    showCloud(x, y, z)
    print(equaSphere(x, y, z))
