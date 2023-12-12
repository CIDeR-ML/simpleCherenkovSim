import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
from matplotlib.colors import LinearSegmentedColormap, Normalize


def point_to_point_dist(p1, p2):
    return np.sqrt(np.sum([(p1[i] - p2[i]) ** 2 for i in range(3)]))


def create_color_gradient(max_cnts, colormap="viridis"):
    # Define the colormap and normalization
    cmap = plt.get_cmap(colormap)
    norm = Normalize(vmin=0, vmax=max_cnts)

    # Create a scalar mappable
    scalar_mappable = plt.cm.ScalarMappable(norm=norm, cmap=cmap)
    scalar_mappable.set_array([])

    return scalar_mappable


def generate_isotropic_random_vector():
    # Generate random azimuthal angle (phi) in the range [0, 2*pi)
    phi = 2 * np.pi * np.random.rand()

    # Generate random polar angle (theta) in the range [0, pi)
    theta = np.arccos(2 * np.random.rand() - 1)

    # Convert spherical coordinates to Cartesian coordinates
    x = np.sin(theta) * np.cos(phi)
    y = np.sin(theta) * np.sin(phi)
    z = np.cos(theta)

    return np.array([x, y, z])


def attenuation_weight(x, L=5):
    return np.exp(-x / L)


def show_2D_display(
    ID_to_position,
    ID_to_PE,
    ID_to_case,
    cyl_sensor_radius,
    cyl_radius,
    cyl_height,
    file_name=None,
):
    max_PE = np.max(list(ID_to_PE.values()))
    color_gradient = create_color_gradient(max_PE)

    fig, ax = plt.subplots(figsize=(8, 8), facecolor="black")
    # fig.patch.set_visible(False)
    # ax.axis('off')

    for ID in list(ID_to_position.keys()):
        pos = ID_to_position[ID]
        PE = ID_to_PE[ID]
        case = ID_to_case[ID]

        caps_offset = 0.05

        # barrel
        if case == 0:
            theta = np.arctan(pos[1] / pos[0]) if pos[0] != 0 else np.pi / 2
            theta += np.pi / 2
            if pos[0] > 0:
                theta += np.pi
            theta /= 2
            z = pos[2] / cyl_height

            ax.add_patch(
                plt.Circle(
                    (theta, z),
                    cyl_sensor_radius / cyl_height,
                    color=color_gradient.to_rgba(PE),
                )
            )

        elif case == 1:
            ax.add_patch(
                plt.Circle(
                    (
                        pos[0] / cyl_height + np.pi / 2,
                        1 + caps_offset + pos[1] / cyl_height,
                    ),
                    cyl_sensor_radius / cyl_height,
                    color=color_gradient.to_rgba(PE),
                )
            )

        elif case == 2:
            ax.add_patch(
                plt.Circle(
                    (
                        pos[0] / cyl_height + np.pi / 2,
                        -1 - caps_offset - pos[1] / cyl_height,
                    ),
                    cyl_sensor_radius / cyl_height,
                    color=color_gradient.to_rgba(PE),
                )
            )

    margin = 0.05
    # plt.gca().set_xlim(-margin,np.pi+margin)
    # plt.gca().set_ylim(-margin-0,1+margin)

    ax.set_facecolor("black")

    # hide x-axis
    ax.get_xaxis().set_visible(False)
    # hide y-axis
    ax.get_yaxis().set_visible(False)
    plt.axis("equal")
    fig.tight_layout()
    if file_name:
        plt.savefig(file_name)
    plt.show()
