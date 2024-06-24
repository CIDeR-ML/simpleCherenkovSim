import time
import pandas as pd
import h5py
import json
import numpy as np
import argparse

import sys, os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from tools.geometry import *

import jax
import jax.numpy as jnp
from jax import value_and_grad
from jax import random
from jax.lax import scan, while_loop


def normalize(R):
    return R / jnp.linalg.norm(R)

def generate_vectors_on_cone_surface_jax(R, theta, num_vectors=10, key=random.PRNGKey(0)):
    """ Generate vectors on the surface of a cone around R. """
    R = normalize(R)
    
    # Generate random azimuthal angles from 0 to 2pi
    phi_values = random.uniform(key, (num_vectors,), minval=0, maxval=2 * jnp.pi)
    
    # Generate vectors in the local coordinate system
    x_local = jnp.sin(theta) * jnp.cos(phi_values)
    y_local = jnp.sin(theta) * jnp.sin(phi_values)
    z_local = jnp.cos(theta) * jnp.ones_like(phi_values)
    
    local_vectors = jnp.stack([x_local, y_local, z_local], axis=-1)
    
    # Compute the rotation matrix to align [0, 0, 1] with R
    v = jnp.cross(jnp.array([0., 0., 1.]), R)
    s = jnp.linalg.norm(v)
    c = R[2]  # dot product of [0, 0, 1] and R
    
    if s == 0:
        # R is already aligned with [0, 0, 1] or its opposite
        rotation_matrix = jnp.eye(3) if c > 0 else jnp.diag(jnp.array([1., 1., -1.]))
    else:
        v_cross = jnp.array([
            [0, -v[2], v[1]],
            [v[2], 0, -v[0]],
            [-v[1], v[0], 0]
        ])
        rotation_matrix = jnp.eye(3) + v_cross + v_cross.dot(v_cross) * (1 - c) / (s ** 2)
    
    # Apply the rotation to all vectors
    rotated_vectors = jnp.dot(local_vectors, rotation_matrix.T)
    
    return rotated_vectors

@jax.jit
def rotate_vector_batch_jax(vectors, axis, angle):
    """ Rotate multiple vectors by specified angle around the given axis. """
    axis_normalized = axis / jnp.linalg.norm(axis)
    
    # Quaternion rotation
    sin_half_angle = jnp.sin(angle / 2)
    cos_half_angle = jnp.cos(angle / 2)
    quaternion = jnp.concatenate([axis_normalized * sin_half_angle, jnp.array([cos_half_angle])])
    
    # Pad vectors to quaternions
    vector_quaternions = jnp.pad(vectors, ((0, 0), (1, 0)))
    
    # Quaternion multiplication
    def quaternion_multiply(q1, q2):
        w1, x1, y1, z1 = jnp.split(q1, 4, axis=-1)
        w2, x2, y2, z2 = jnp.split(q2, 4, axis=-1)
        return jnp.concatenate([
            w1*x2 + x1*w2 + y1*z2 - z1*y2,
            w1*y2 - x1*z2 + y1*w2 + z1*x2,
            w1*z2 + x1*y2 - y1*x2 + z1*w2,
            w1*w2 - x1*x2 - y1*y2 - z1*z2
        ], axis=-1)
    
    rotated_quaternions = quaternion_multiply(
        quaternion_multiply(quaternion, vector_quaternions),
        jnp.concatenate([-axis_normalized * sin_half_angle, jnp.array([cos_half_angle])])
    )
    
    return rotated_quaternions[..., 1:]


def test_generate_vectors_on_cone_surface():
    # Test parameters
    R = jnp.array([0., 0., 1.])  # Cone axis along z-axis for simplicity
    theta = jnp.radians(30.)  # 30 degree cone opening
    num_vectors = 10000
    
    # Generate vectors
    vectors = generate_vectors_on_cone_surface_jax(R, theta, num_vectors)
    
    # Test 1: Check angles
    dot_products = jnp.dot(vectors, R)
    angles = jnp.arccos(dot_products / (jnp.linalg.norm(vectors, axis=1) * jnp.linalg.norm(R)))
    mean_angle = jnp.mean(angles)
    angle_std = jnp.std(angles)
    print(f"Mean angle: {jnp.degrees(mean_angle):.2f} degrees (should be close to 30)")
    print(f"Angle standard deviation: {jnp.degrees(angle_std):.2f} degrees (should be small)")
    
    # Test 2: Check uniform distribution
    xy_projection = vectors[:, :2]
    xy_angles = jnp.arctan2(xy_projection[:, 1], xy_projection[:, 0])
    hist, _ = jnp.histogram(xy_angles, bins=36, range=(-jnp.pi, jnp.pi))
    uniformity = jnp.std(hist) / jnp.mean(hist)
    print(f"Uniformity measure: {uniformity:.4f} (should be close to 0)")
    
    # Test 3: Check magnitudes
    magnitudes = jnp.linalg.norm(vectors, axis=1)
    mean_magnitude = jnp.mean(magnitudes)
    magnitude_std = jnp.std(magnitudes)
    print(f"Mean magnitude: {mean_magnitude:.4f} (should be close to 1)")
    print(f"Magnitude standard deviation: {magnitude_std:.4f} (should be small)")

# Call the test function
test_generate_vectors_on_cone_surface()