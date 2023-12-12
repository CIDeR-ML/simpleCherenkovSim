import numpy as np


class Ray:  # a basic object corresponding to a line with an origin point, a direction vector and a sensor of motion.
    def __init__(self, origin, vector):
        self.O = origin
        self.V = vector
