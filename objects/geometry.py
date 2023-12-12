import numpy as np


class Cylinder:
    def __init__(
        self, center, axis, radius, height, barrel_grid, cap_rings, cyl_sensor_radius
    ):
        self.C = center
        self.A = axis
        self.r = radius
        self.H = height
        self.S_radius = cyl_sensor_radius

        self.place_photosensors(barrel_grid, cap_rings)

    def ray_cylinder_intersection(self, ray_origin, ray_direction):
        # Ray: P = O + tD, where P is a point on the ray, O is the ray origin, D is the ray direction, and t is a parameter.
        # Cylinder: (P - C - ((P - C) dot A)A)^2 = r^2, where C is the cylinder center, A is the cylinder axis, and r is the radius.

        print("--------")
        intersection_point1 = 0
        intersection_point2 = 0

        # Define the variables
        O = np.array(ray_origin)
        D = np.array(ray_direction)
        C = self.C
        A = self.A
        r = self.r
        H = self.H

        # Calculate coefficients for the quadratic equation
        a = np.dot(D - np.dot(D, A) * A, D - np.dot(D, A) * A)
        b = 2 * np.dot(D - np.dot(D, A) * A, O - C - np.dot(O - C, A) * A)
        c = np.dot(O - C - np.dot(O - C, A) * A, O - C - np.dot(O - C, A) * A) - r**2

        # Solve the quadratic equation
        discriminant = b**2 - 4 * a * c

        if a >= 0:
            # Calculate the solutions for t (parameter along the ray)
            t1 = (-b + np.sqrt(discriminant)) / (2 * a)
            t2 = (-b - np.sqrt(discriminant)) / (2 * a)

            # Check if the intersection points are in the positive direction of the ray
            intersection_point1 = (
                O + t1 * D
            )  # if t1 >= 0 and 0 <= (O + t1 * D - C).dot(A) <= H/2 else None
            intersection_point2 = (
                O + t2 * D
            )  # if t2 >= 0 and 0 <= (O + t2 * D - C).dot(A) <= H/2 else None

            # print(intersection_point1, intersection_point2)

            # if intersection_point1 is not None:
            #     return intersection_point1

            # if intersection_point2 is not None:
            #     return intersection_point2

            # print('aaaa:' ,intersection_point1, intersection_point2)

        # If no intersection with the barrel, check for intersection with the caps
        t_cap1 = (H / 2 - (O - C).dot(A)) / D.dot(A) if D.dot(A) != 0 else np.inf
        t_cap2 = -((O - C).dot(A)) / D.dot(A) if D.dot(A) != 0 else np.inf

        cap_intersection1 = O + abs(t_cap1) * D
        cap_intersection2 = O + abs(t_cap2) * D

        # print('cap int: ',cap_intersection1, cap_intersection2, np.sum(cap_intersection1), np.sum(cap_intersection2))

        # if np.sum(cap_intersection1) == 0:
        #     return cap_intersection2

        # if np.sum(cap_intersection2) == 0:
        #     return cap_intersection1

        print()

        # assert t_cap1 > 0 and t_cap2 > 0

        # if t_cap1 > 0:
        #     return cap_intersection2

        # if t_cap2 > 0:
        #     return cap_intersection1

        # print(t_cap1, t_cap2, cap_intersection1, cap_intersection2)
        return None

    def distance_ray_point_vectorized(self, ray_origin, ray_direction, points):
        # Convert inputs to NumPy arrays
        ray_origin = np.array(ray_origin)
        ray_direction = np.array(ray_direction)
        points = np.array(points)

        # Calculate vectors from ray origin to all points
        vectors_to_points = points - ray_origin

        # Project all vectors onto the ray direction
        t_values = np.dot(vectors_to_points, ray_direction) / np.dot(
            ray_direction, ray_direction
        )

        # Calculate the points on the ray closest to the given points
        closest_points_on_ray = ray_origin + t_values[:, np.newaxis] * ray_direction

        # Calculate the Euclidean distances between all points and their closest points on the ray
        distances = np.linalg.norm(points - closest_points_on_ray, axis=1)

        return distances

    def distance_ray_point(self, ray_origin, ray_direction, point):
        # Calculate the vector from ray origin to the point
        vector_to_point = np.array(point) - np.array(ray_origin)

        # Project the vector onto the ray direction
        t = np.dot(vector_to_point, ray_direction) / np.dot(
            ray_direction, ray_direction
        )

        # Calculate the point on the ray closest to the given point
        closest_point_on_ray = np.array(ray_origin) + t * np.array(ray_direction)

        # Calculate the Euclidean distance between the two points
        distance = np.linalg.norm(np.array(point) - closest_point_on_ray)

        return distance

    # def check_dir(self, ray_origin, ray_direction, point):
    #     # Calculate distances from the ray origin to each point along the ray direction
    #     distance_to_point = np.dot(np.array(point) - np.array(ray_origin), ray_direction)
    #     return distance_to_point >= 0

    # def search_nearest_sensor_for_ray(self, ray_origin, ray_direction):
    #     #distances  = [self.distance_ray_point(ray_origin,ray_direction,p) for p in self.all_points]
    #     distances = self.distance_ray_point_vectorized(ray_origin, ray_direction, self.all_points)
    #     indices    = np.argsort(distances)

    #     found_idx = None
    #     for i in indices:
    #         if self.check_dir(ray_origin, ray_direction, self.all_points[i]):
    #             found_idx = i
    #             break

    #     return self.all_points[found_idx], distances[found_idx], found_idx

    def check_dir(self, ray_origin, ray_direction, point):
        # Calculate distances from the ray origin to each point along the ray direction
        distance_to_point = np.dot(
            np.array(point) - np.array(ray_origin), ray_direction
        )
        return distance_to_point >= 0

    def search_nearest_sensor_for_ray(self, ray_origin, ray_direction):
        # distances  = [self.distance_ray_point(ray_origin,ray_direction,p) for p in self.all_points]
        distances = self.distance_ray_point_vectorized(
            ray_origin, ray_direction, self.all_points
        )
        distance_to_point = np.dot(
            np.array(self.all_points) - np.array(ray_origin), ray_direction
        )
        # found_idx = np.where(np.min(distances[distance_to_point > 0])==distances)[0][0]

        # Filter out distances corresponding to points in the opposite direction of the ray
        valid_distances = distances[distance_to_point > 0]

        # Find the index of the minimum distance
        min_distance_index = np.argmin(valid_distances)

        # Find the overall index in the original array
        found_idx = np.where(distance_to_point > 0)[0][min_distance_index]

        return self.all_points[found_idx], distances[found_idx], found_idx

    def place_photosensors(self, barrel_grid, cap_rings):
        # barrel ----
        b_rows = barrel_grid[0]
        b_cols = barrel_grid[1]

        theta = np.linspace(
            0, 2 * np.pi, b_cols, endpoint=False
        )  # Generate N angles from 0 to 2pi
        x = self.r * np.cos(theta) + self.C[0]
        y = self.r * np.sin(theta) + self.C[1]
        z = [
            (i + 1) * self.H / (b_rows + 1) - self.H / 2 + self.C[2]
            for i in range(b_rows)
        ]

        barr_points = np.array(
            [[x[j], y[j], z[i]] for i in range(b_rows) for j in range(b_cols)]
        )
        self.barr_points = barr_points

        del x, y, z, theta  # ensure no values are passed to the caps.
        # -----------

        # caps ----
        Nrings = len(cap_rings)

        tcap_points = []
        bcap_points = []
        for i_ring, N_sensors_in_ring in enumerate(cap_rings):
            theta = np.linspace(
                0, 2 * np.pi, N_sensors_in_ring, endpoint=False
            )  # Generate N angles from 0 to 2pi
            x = self.r * ((Nrings - (i_ring + 1)) / Nrings) * np.cos(theta) + self.C[0]
            y = self.r * ((Nrings - (i_ring + 1)) / Nrings) * np.sin(theta) + self.C[1]
            top_z = [self.H / 2 + self.C[2] for i in range(N_sensors_in_ring)]
            bot_z = [-self.H / 2 + self.C[2] for i in range(N_sensors_in_ring)]

            for i_sensor in range(N_sensors_in_ring):
                tcap_points.append([x[i_sensor], y[i_sensor], top_z[i_sensor]])
                bcap_points.append([x[i_sensor], y[i_sensor], bot_z[i_sensor]])

        self.tcap_points = np.array(tcap_points)
        self.bcap_points = np.array(bcap_points)

        self.all_points = np.concatenate(
            [self.barr_points, self.tcap_points, self.bcap_points], axis=0
        )

        # let's make this generic format... ID to 3D pos dictionary
        self.ID_to_position = {
            i: self.all_points[i] for i in range(len(self.all_points))
        }

        self.ID_to_case = {}
        Nbarr = len(self.barr_points)
        Ntcap = len(self.tcap_points)
        Nbcap = len(self.bcap_points)
        for i in range(len(self.all_points)):
            if i < Nbarr:
                self.ID_to_case[i] = 0
            elif Nbarr <= i < Ntcap + Nbarr:
                self.ID_to_case[i] = 1
            elif Ntcap + Nbarr <= i < Nbcap + Ntcap + Nbarr:
                self.ID_to_case[i] = 2
            else:
                print(
                    "check: place_photosensors! this should not be happening: ",
                    Nbarr,
                    Ntcap,
                    Nbcap,
                    i,
                )

        # -----------
