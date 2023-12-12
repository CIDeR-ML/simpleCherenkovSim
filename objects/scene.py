import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
from matplotlib.patches import Circle
import mpl_toolkits.mplot3d.art3d as art3d
from matplotlib.colors import LinearSegmentedColormap, Normalize


def create_color_gradient(max_cnts, colormap="gnuplot"):
    # Define the colormap and normalization
    cmap = plt.get_cmap(colormap)
    norm = Normalize(vmin=0, vmax=max_cnts)

    # Create a scalar mappable
    scalar_mappable = plt.cm.ScalarMappable(norm=norm, cmap=cmap)
    scalar_mappable.set_array([])

    return scalar_mappable


def rotation_matrix(d):
    """
    Calculates a rotation matrix given a vector d. The direction of d
    corresponds to the rotation axis. The length of d corresponds to
    the sin of the angle of rotation.

    Variant of: http://mail.scipy.org/pipermail/numpy-discussion/2009-March/040806.html
    """
    sin_angle = np.linalg.norm(d)

    if sin_angle == 0:
        return np.identity(3)

    d /= sin_angle

    eye = np.eye(3)
    ddt = np.outer(d, d)
    skew = np.array(
        [[0, d[2], -d[1]], [-d[2], 0, d[0]], [d[1], -d[0], 0]], dtype=np.float64
    )

    M = ddt + np.sqrt(1 - sin_angle**2) * (eye - ddt) + sin_angle * skew
    return M


def pathpatch_2d_to_3d(pathpatch, z=0, normal="z"):
    """
    Transforms a 2D Patch to a 3D patch using the given normal vector.

    The patch is projected into they XY plane, rotated about the origin
    and finally translated by z.
    """
    if type(normal) is str:  # Translate strings to normal vectors
        index = "xyz".index(normal)
        normal = np.roll((1.0, 0, 0), index)

    normal /= np.linalg.norm(normal)  # Make sure the vector is normalised

    path = pathpatch.get_path()  # Get the path and the associated transform
    trans = pathpatch.get_patch_transform()

    path = trans.transform_path(path)  # Apply the transform

    pathpatch.__class__ = art3d.PathPatch3D  # Change the class
    pathpatch._code3d = path.codes  # Copy the codes
    pathpatch._facecolor3d = pathpatch.get_facecolor  # Get the face color

    verts = path.vertices  # Get the vertices in 2D

    d = np.cross(normal, (0, 0, 1))  # Obtain the rotation vector
    M = rotation_matrix(d)  # Get the rotation matrix

    pathpatch._segment3d = np.array(
        [np.dot(M, (x, y, 0)) + (0, 0, z) for x, y in verts]
    )


def pathpatch_translate(pathpatch, delta):
    """
    Translates the 3D pathpatch by the amount delta.
    """
    pathpatch._segment3d += delta


class Scene:  # a basic object to plot 3D things
    def __init__(self, x_size=8, y_size=8):
        self.fig = plt.figure(figsize=(x_size, y_size))
        self.ax = self.fig.add_subplot(111, projection="3d")

    def plot_3d_circle_surface(
        self, origin, parallel_vector, normal_vector, r, c="k", num_points=100
    ):
        p = Circle((0, 0), r, alpha=1.0, facecolor=c)
        self.ax.add_patch(p)
        if np.dot(normal_vector, [0, 0, 1]) != 1:
            pathpatch_2d_to_3d(p, z=0, normal=normal_vector)
        else:
            pathpatch_2d_to_3d(p, z=0, normal="z")
        pathpatch_translate(p, (origin[0], origin[1], origin[2]))

    def plot_2d_circle_in_3d(
        self, origin, parallel_vector, normal_vector, r, c="k", num_points=100
    ):
        # Generate points on the circle
        theta = np.linspace(0, 2 * np.pi, num_points)
        # Compute the cross product (i x v)
        parallel_vector = parallel_vector / np.linalg.norm(parallel_vector)
        cross_product = np.cross(parallel_vector, normal_vector)
        cross_product = cross_product / np.linalg.norm(cross_product)

        # Parametric equations for the 3D circle
        x = origin[0] + r * (
            np.cos(theta) * cross_product[0] + np.sin(theta) * parallel_vector[0]
        )
        y = origin[1] + r * (
            np.cos(theta) * cross_product[1] + np.sin(theta) * parallel_vector[1]
        )
        z = origin[2] + r * (
            np.cos(theta) * cross_product[2] + np.sin(theta) * parallel_vector[2]
        )

        self.ax.plot(x, y, z, lw=1, c=c)

    def add_point(self, p, c="k"):
        self.ax.scatter(p[0], p[1], p[2], c=c)

    def add_ray(self, p, v, L, c="k"):
        t = np.linspace(0, L, 100)
        ray_points = np.array([p + ti * v for ti in t])
        x = ray_points[:, 0]
        y = ray_points[:, 1]
        z = ray_points[:, 2]
        self.ax.plot(x, y, z, c)

    def add_photosensors(
        self, det, colors=["k", "k", "k"], use_circ=True, use_points=False
    ):
        for i, p_list in enumerate([det.barr_points, det.tcap_points, det.bcap_points]):
            x = p_list[:, 0]
            y = p_list[:, 1]
            z = p_list[:, 2]

            if use_points:
                self.ax.scatter(p[0], p[1], p[2], c=c)

            if use_circ:
                for _x, _y, _z in zip(x, y, z):
                    if i == 0:  # barrel
                        self.plot_3d_circle_surface(
                            [_x, _y, _z],
                            det.A,
                            [_x, _y, 0] / np.linalg.norm([_x, _y, 0]),
                            r=det.S_radius,
                            c=colors[i],
                        )
                        self.plot_2d_circle_in_3d(
                            [_x, _y, _z],
                            det.A,
                            [_x, _y, 0] / np.linalg.norm([_x, _y, 0]),
                            r=det.S_radius,
                            c=colors[i],
                        )
                    else:  # caps
                        self.plot_3d_circle_surface(
                            [_x, _y, _z], [1, 0, 0], det.A, det.S_radius, c=colors[i]
                        )
                        self.plot_2d_circle_in_3d(
                            [_x, _y, _z], [1, 0, 0], det.A, det.S_radius, c=colors[i]
                        )

    def add_photocounts(
        self, det, counts_per_sensor, show_all=True, colors=["k", "k", "k"]
    ):
        max_cnts = np.max(list(counts_per_sensor.values()))
        print(max_cnts)
        color_gradient = create_color_gradient(max_cnts)

        j = 0
        for i, p_list in enumerate([det.barr_points, det.tcap_points, det.bcap_points]):
            x = p_list[:, 0]
            y = p_list[:, 1]
            z = p_list[:, 2]

            for _x, _y, _z in zip(x, y, z):
                if show_all or counts_per_sensor[j]:
                    if i == 0:  # barrel
                        self.plot_3d_circle_surface(
                            [_x, _y, _z],
                            det.A,
                            [_x, _y, 0] / np.linalg.norm([_x, _y, 0]),
                            r=det.S_radius,
                            c=color_gradient.to_rgba(counts_per_sensor[j]),
                        )
                        self.plot_2d_circle_in_3d(
                            [_x, _y, _z],
                            det.A,
                            [_x, _y, 0] / np.linalg.norm([_x, _y, 0]),
                            r=det.S_radius,
                            c=colors[i],
                        )
                    else:  # caps
                        self.plot_3d_circle_surface(
                            [_x, _y, _z],
                            [1, 0, 0],
                            det.A,
                            det.S_radius,
                            c=color_gradient.to_rgba(counts_per_sensor[j]),
                        )
                        self.plot_2d_circle_in_3d(
                            [_x, _y, _z], [1, 0, 0], det.A, det.S_radius, c=colors[i]
                        )

                j += 1

    def show(self, det=None):
        if det:
            plt.gca().set_xlim(-det.r * 1.2, det.r * 1.2)
            plt.gca().set_ylim(-det.r * 1.2, det.r * 1.2)
            plt.gca().set_zlim(-det.H * 0.6, det.H * 0.6)
        plt.show()
