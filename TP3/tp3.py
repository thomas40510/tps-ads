import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as cl
from mpl_toolkits.mplot3d import axes3d


def dataset(file: str):
    res = np.load(file)
    return res


def shapeIt(D):
    x = D[:, 0]
    y = D[:, 1]
    z = D[:, 2]

    x = x.reshape(200, 200)
    y = y.reshape(200, 200)
    z = z.reshape(200, 200)
    return x, y, z


def cloud():
    D = dataset("data/sein.npy")
    x = D[:, 0]
    y = D[:, 1]
    z = D[:, 2]
    plt.scatter(x, y, c=z, edgecolors='none',
                cmap='gist_earth', vmin=-15, vmax=15)
    plt.colorbar()
    plt.show()


def surface():
    D = dataset("data/sein.npy")
    x, y, z = shapeIt(D)

    plt.pcolormesh(x, y, z, cmap='gist_earth', shading='auto',
                   vmin=-15, vmax=15)
    plt.colorbar()
    plt.contour(x, y, z, levels=5, vmin=-15, vmax=15, colors='black')
    plt.title("Surface et lignes de niveaux")
    plt.show()


def light():
    D = dataset("data/sein.npy")
    x, y, z = shapeIt(D)

    L = cl.LightSource(azdeg=45, altdeg=35)
    S = L.shade(z, cmap=plt.get_cmap('gist_earth'),
                vmin=-15, vmax=15,
                blend_mode='soft', vert_exag=5)
    plt.imshow(S)
    plt.show()


a = 6.4


def sphere():
    lbda, phi = np.meshgrid(np.linspace(0, 2 * np.pi, 100),
                            np.linspace(-np.pi / 2, np.pi / 2, 100))
    # phi = np.meshgrid(np.linspace(-np.pi / 2, np.pi / 2, 100))

    x = a * np.cos(phi) * np.cos(lbda)
    y = a * np.cos(phi) * np.sin(lbda)
    z = a * np.sin(phi)

    return x, y, z


def rdmSphere(lbda, phi):
    x = a * np.cos(phi) * np.cos(lbda)
    y = a * np.cos(phi) * np.sin(lbda)
    z = a * np.sin(phi)

    return x, y, z


def showSphere():
    x, y, z = sphere()
    fig = plt.figure()
    ax = fig.gca(projection='3d')
    # ax.plot_wireframe(x, y, z)
    ax.plot_surface(x, y, z)
    plt.show()


def earth():
    npz = np.load('data/world.npz')
    lat = npz['lat'] * np.pi / 180
    long = npz['lon'] * np.pi / 180
    elev = npz['elev']
    lbda, phi = np.meshgrid(long, lat)
    x, y, z = rdmSphere(lbda, phi)

    emin, emax = np.min(elev), np.max(elev)
    # elev_n = elev / (emax - emin)
    elev_n = (elev-elev[:].min())/(elev[:].max()-elev[:].min())

    fig = plt.figure()
    ax = fig.gca(projection='3d')
    ax.plot_surface(x, y, z,
                    facecolors=plt.get_cmap('gist_earth')(elev_n),
                    cstride=2, rstride=2,
                    linewidth=0, antialiased=False)
    plt.show()


if __name__ == '__main__':
    # cloud()
    # surface()
    # light()
    # showSphere()
    earth()
