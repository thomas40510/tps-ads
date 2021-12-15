import numpy as np
import matplotlib.pyplot as plt


def readDataset(file):
    f = open(file)
    rawData = f.readlines()
    f.close()
    rawData.pop(0)
    res = []
    for l in rawData:
        tmp = l.replace("\n", "")
        splt = [float(i) for i in tmp.split(" ")]
        res.append(splt)
    return np.asarray(res)


def nuage_points(dataset):
    """
    Affiche les profondeurs sous forme de nuage de points
    :param dataset: Liste des profondeurs mesurées
    """
    X = np.arange(0, len(dataset), 1)
    plt.scatter(X, dataset)
    plt.xlabel("Numéro d'enregistrement")
    plt.ylabel("Profondeur")
    plt.title("Relevés de profondeur")
    plt.show()


def cuir_moustache(dataset):
    """
    Affiche les profondeurs sous forme de boîtes à moustaches
    :param dataset: liste des profondeurs mesurées
    """
    f, ax = plt.subplots(1, 4, sharey='all')
    ax[0].boxplot(dataset)
    ax[0].set_title("par défaut")
    ax[1].boxplot(dataset, whis=.7)
    ax[1].set_title("plus courte")
    c = 'b'
    ax[2].boxplot(dataset,
                  flierprops=dict(color=c, markeredgecolor=c))
    ax[2].set_title("coloré")
    ax[3].boxplot(dataset, showmeans=True)
    ax[3].set_title("avec la moyenne")
    plt.show()


def cumul(dataset):
    """
    Affiche les profondeurs cumulées
    :param dataset: liste des mesures de profondeur
    """
    ds = np.copy(dataset)
    ds.sort()
    X = np.linspace(0, 1, len(dataset))
    plt.plot(ds, X)
    plt.show()


def histo(dataset):
    ds = np.copy(dataset)
    ds.sort()
    L = np.arange(0, len(ds), int(len(ds) / 19))
    C = ds[L]
    print(C)
    print(L)
    f, ax = plt.subplots(2, 1, sharex='all')
    ax[0].hist(ds, bins=20, density=True)
    ax[0].set_xlabel("largeur constante")
    ax[0].set_ylabel('Densité')
    # ax[1].hist(bins[:-1], bins, weights=counts, density=True)
    ax[1].hist(ds, bins=C, density=True)
    ax[1].set_xlabel("effectifs constants")
    ax[1].set_ylabel('Densité')
    plt.show()


def minMax(lst):
    return min(lst), max(lst)


def splitLists(lst, dataset, n):
    res = []
    step = int(len(lst) / n)
    for i in range(n):
        res.append(dataset[(i - 1) * step:i * step])
    res.pop(0)
    return np.asarray(res)


def splitLst(lst, dataset, n):
    res = []
    step = int(len(lst) / n)
    for i in range(n):
        res.append([lst[(i - 1) * step:i * step], dataset[(i - 1) * step:i * step]])
    res.pop(0)
    return np.asarray(res)


def splitSplitList(N, E, prof, n):
    res = []
    stp = (int(len(N) / n), int(len(E) / n))
    for i in range(1, n):
        tmp = []
        for j in range(1, n):
            step = stp[1]
            tmp.append([E[(j - 1) * step:j * step], prof[(j - 1) * step:j * step]])
        step = stp[0]
        res.append([N[(i - 1) * step:i * step], tmp])
    return res


def cuirCuirMoustache(dataset):
    plt.boxplot(dataset, positions=np.arange(0, len(dataset), 1))
    plt.xlabel("Numéro de tranche en Easting")
    plt.ylabel("Profondeur")
    plt.title("Profondeurs par tranches de latitudes")
    plt.show()


def quantiles(dataset):
    X = dataset[:, 0, 0]
    med = [np.median(dataset[i, 1, :]) for i in range(len(dataset))]
    mean = [np.mean(dataset[i, 1, :]) for i in range(len(dataset))]
    q1 = [np.quantile(dataset[i, 1, :], .25) for i in range(len(dataset))]
    q3 = [np.quantile(dataset[i, 1, :], .75) for i in range(len(dataset))]
    sigma = [np.std(dataset[i, 1, :]) for i in range(len(dataset))]

    qt1 = [np.quantile(dataset[i, 1, :], 1 / 40) for i in range(len(dataset))]
    qt39 = [np.quantile(dataset[i, 1, :], 39 / 40) for i in range(len(dataset))]

    Conf1 = np.asarray(mean) - 2 * np.asarray(sigma)
    Conf2 = np.asarray(mean) + 2 * np.asarray(sigma)

    plt.scatter(X, med, label='médiane')
    plt.scatter(X, mean, label='moyenne $\mu$')
    plt.scatter(X, q1, label='$Q1$')
    plt.scatter(X, q3, label='$Q3$')
    plt.scatter(X, qt1, label='$Q_t1$')
    plt.scatter(X, qt39, label='$Q_t39$')
    plt.plot(X, Conf1, label='$\mu - 2\sigma$')
    plt.plot(X, Conf2, label='$\mu + 2\sigma$')
    plt.xlabel("Easting")
    plt.ylabel("profondeur")
    plt.legend()
    plt.title("Données statistiques sur les classes")
    plt.show()


def histo2d(north, east, prof):
    f, ax = plt.subplots()
    h = ax.hist2d(east, north, bins=40, density=False, cmin=10)
    f.colorbar(h[3], ax=ax)
    plt.xlabel("Easting")
    plt.ylabel("Northing")
    plt.show()


if __name__ == '__main__':
    ds = readDataset("data/Processed.3d")
    north = ds[:, 0]
    east = ds[:, 1]
    prof = ds[:, 2]
    # nuage_points(ds[:, 2])
    # cuir_moustache(prof)
    # cumul(prof)
    # histo(prof)
    # dt = splitLst(east, prof, 20)
    # cuirCuirMoustache(dt)
    # quantiles(dt)
    # histo2d(north, east, prof)

    dt = splitSplitList(north, east, prof, 20)
    print(dt)
