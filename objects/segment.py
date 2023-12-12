import numpy as np

# def sample_cone_surface(num_samples, cone_height=2.0, cone_radius=3.0, cone_axis=np.array([0, 0, 1])):
#     # Generate random cylindrical coordinates
#     theta = 2 * np.pi * np.random.rand(num_samples)
#     z = cone_height * np.random.rand(num_samples)
#     r = cone_radius * (1 - z / cone_height)

#     # Convert cylindrical coordinates to Cartesian coordinates
#     x = r * np.cos(theta)
#     y = r * np.sin(theta)
#     z = z

#     # Rotate the cone points to align with the arbitrary axis
#     rotation_matrix = get_rotation_matrix(cone_axis)
#     points = np.vstack((x, y, z))
#     rotated_points = np.dot(rotation_matrix, points)

#     print(np.dot(rotation_matrix, [0,0,cone_height]), cone_height)

#     #return np.array([rotated_points[0], rotated_points[1], rotated_points[2]])
#     return np.array([points[0], points[1], points[2]-4])

# def get_rotation_matrix(axis):
#     axis = axis / np.linalg.norm(axis)
#     angle = np.arccos(np.dot(axis, np.array([0, 0, 1])))
#     cross = np.cross(np.array([0, 0, 1]), axis)
#     cross /= np.linalg.norm(cross)
#     rotation_matrix = np.array([[np.cos(angle) + cross[0]**2 * (1 - np.cos(angle)),
#                                  cross[0] * cross[1] * (1 - np.cos(angle)) - cross[2] * np.sin(angle),
#                                  cross[0] * cross[2] * (1 - np.cos(angle)) + cross[1] * np.sin(angle)],
#                                 [cross[1] * cross[0] * (1 - np.cos(angle)) + cross[2] * np.sin(angle),
#                                  np.cos(angle) + cross[1]**2 * (1 - np.cos(angle)),
#                                  cross[1] * cross[2] * (1 - np.cos(angle)) - cross[0] * np.sin(angle)],
#                                 [cross[2] * cross[0] * (1 - np.cos(angle)) - cross[1] * np.sin(angle),
#                                  cross[2] * cross[1] * (1 - np.cos(angle)) + cross[0] * np.sin(angle),
#                                  np.cos(angle) + cross[2]**2 * (1 - np.cos(angle))]])
#     return rotation_matrix


def rotate_vector(vector, axis, angle):
    """Rotate a vector around an axis by a given angle in radians."""
    axis = normalize(axis)
    cos_angle = np.cos(angle)
    sin_angle = np.sin(angle)
    cross_product = np.cross(axis, vector)
    dot_product = np.dot(axis, vector) * (1 - cos_angle)
    return cos_angle * vector + sin_angle * cross_product + dot_product * axis


def normalize(v):
    """Normalize a vector."""
    norm = np.linalg.norm(v)
    if norm == 0:
        return v
    return v / norm


def generate_vectors_on_cone_surface(R, theta, num_vectors=10):
    """Generate vectors on the surface of a cone around R."""
    R = normalize(R)
    vectors = []

    for _ in range(num_vectors):
        # Random azimuthal angle from 0 to 2pi
        phi = np.random.uniform(0, 2 * np.pi)

        # Spherical to Cartesian coordinates in the local system
        x = np.sin(theta) * np.cos(phi)
        y = np.sin(theta) * np.sin(phi)
        z = np.cos(theta)

        local_vector = np.array([x, y, z])

        # Find rotation axis and angle to align local z-axis with R
        z_axis = np.array([0, 0, 1])
        axis = np.cross(z_axis, R)
        if np.linalg.norm(axis) != 0:  # If R is not already along z-axis
            angle = np.arccos(np.dot(z_axis, R))
            local_vector = rotate_vector(local_vector, axis, angle)

        vectors.append(local_vector)

    return np.array(vectors)


class Segment:  # a basic object corresponding to a line with an origin point, a direction vector and a sensor of motion.
    def __init__(self, origin, vector, length):
        self.O = origin
        self.V = vector
        self.L = length
        self.rays = []

    def gen_ch_rays(self, angle=40, num_samples=1000):
        # Sample cone surface
        self.ray_vectors = generate_vectors_on_cone_surface(self.V, angle, num_samples)
