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

from functools import partial


@jax.jit
def normalize(vector):
    return vector / jnp.linalg.norm(vector)

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

def check_hits_vectorized_per_track_jax(ray_origin, ray_direction, sensor_radius, points):
    # Ensure inputs are JAX arrays
    ray_origin_jax = jnp.array(ray_origin, dtype=jnp.float32)
    ray_direction_jax = jnp.array(ray_direction, dtype=jnp.float32)
    points_jax = jnp.array(points, dtype=jnp.float32)

    # Calculate vectors from ray origin to all points
    vectors_to_points = points_jax - ray_origin_jax[:, None, :]

    # Project all vectors onto the ray direction
    dot_products_numerator = jnp.einsum('ijk,ik->ij', vectors_to_points, ray_direction_jax)
    dot_products_denominator = jnp.sum(ray_direction_jax * ray_direction_jax, axis=-1)

    # Calculate t_values
    t_values = dot_products_numerator / dot_products_denominator[:, None]

    # Calculate the points on the ray closest to the given points
    closest_points_on_ray = ray_origin_jax[:, None, :] + t_values[:, :, None] * ray_direction_jax[:, None, :]

    # Calculate the Euclidean distances between all points and their closest points on the ray
    distances = jnp.linalg.norm(points_jax - closest_points_on_ray, axis=2)

    # Apply the mask
    mask = t_values < 0
    distances = jnp.where(mask, 999.0, distances)

    # Find the indices of the minimum distances
    indices = jnp.argmin(distances, axis=1)

    # True if the photon is on the photosensor False otherwise
    hit_flag = distances[jnp.arange(indices.size), indices] < sensor_radius

    # Get the good indices based on sensor_radius
    sensor_indices = indices[hit_flag]

    return sensor_indices, hit_flag, closest_points_on_ray

def generate_data(json_filename, output_filename, cone_opening, track_origin, track_direction):
    # Generate detector (photsensor placements)
    detector = generate_detector(json_filename)
    N_photosensors = len(detector.all_points)

    print('Generating data dataset')

    Nhits = 0 # this is a counter used to keep track on how many hits we have filled in total for every event.
    Nevents = 1
    Ntrk    = 1

    # Create the output h5 file and define some fields we can start filling.
    f_outfile = h5py.File(output_filename, 'w')
    h5_evt_ids     = f_outfile.create_dataset("evt_id",           shape=(Nevents,),    dtype=np.int32)
    h5_evt_pos     = f_outfile.create_dataset("positions",        shape=(Nevents,1,3), dtype=np.float32)
    h5_evt_hit_idx = f_outfile.create_dataset("event_hits_index", shape=(Nevents,),    dtype=np.int64)

    f_outfile.create_dataset("true_cone_opening", data=np.array([cone_opening]))
    f_outfile.create_dataset("true_track_origin", data=track_origin)
    f_outfile.create_dataset("true_track_direction", data=track_direction)

    maxNhits = Nevents * N_photosensors
    h5_evt_hit_IDs_max = np.zeros(maxNhits)
    h5_evt_hit_Qs_max  = np.zeros(maxNhits)
    h5_evt_hit_Ts_max  = np.zeros(maxNhits)

    pre_idx = 0
    i_evt = 0

    Nphot = 100
    ray_vectors, ray_origins = get_rays(track_origin, track_direction, cone_opening, Nphot)

    sensor_indices, _, _ = check_hits_vectorized_per_track_jax(np.array(ray_origins, dtype=np.float32),\
                                                       np.array(ray_vectors, dtype=np.float32), \
                                                       detector.S_radius, \
                                                       np.array(detector.all_points,dtype=np.float32))

    idx, cts = np.unique(sensor_indices, return_counts=True)
    Nhits += len(idx)
    h5_evt_hit_idx[i_evt] = Nhits
    h5_evt_hit_IDs_max[pre_idx:Nhits] = idx
    h5_evt_hit_Qs_max [pre_idx:Nhits] = cts
    h5_evt_hit_Ts_max [pre_idx:Nhits] = np.zeros(len(cts))
    pre_idx = Nhits

    h5_evt_hit_IDs = f_outfile.create_dataset("hit_pmt",          shape=(Nhits,),      dtype=np.int32)
    h5_evt_hit_Qs  = f_outfile.create_dataset("hit_charge",       shape=(Nhits,),      dtype=np.float32)
    h5_evt_hit_Ts  = f_outfile.create_dataset("hit_time",         shape=(Nhits,),      dtype=np.float32)

    h5_evt_hit_IDs[0:Nhits] = h5_evt_hit_IDs_max[0:Nhits]
    h5_evt_hit_Qs[0:Nhits]  = h5_evt_hit_Qs_max [0:Nhits]
    h5_evt_hit_Ts[0:Nhits]  = h5_evt_hit_Ts_max [0:Nhits]

    print(sensor_indices)

    f_outfile.close()
    print('Data generation complete.')

def load_data(filename):
    with h5py.File(filename, 'r') as f:
        hit_pmt = np.array(f['hit_pmt'])
        hit_charge = np.array(f['hit_charge'])
        hit_time = np.array(f['hit_time'])
        true_cone_opening = np.array(f['true_cone_opening'])[0]
        true_track_origin = np.array(f['true_track_origin'])
        true_track_direction = np.array(f['true_track_direction'])
    return hit_pmt, hit_charge, hit_time, true_cone_opening, true_track_origin, true_track_direction

def get_rays(track_origin, track_direction, cone_opening, Nphot):

    ray_vectors = generate_vectors_on_cone_surface_jax(track_direction, jnp.radians(cone_opening), Nphot)
    ray_origins = jnp.ones((Nphot, 3)) * track_origin + np.random.uniform(0, 1, (Nphot, 1))*track_direction

    return ray_vectors, ray_origins

@partial(jax.jit, static_argnums=(3,))
def differentiable_get_rays(track_origin, track_direction, cone_opening, Nphot, key):
    key, subkey = random.split(key)
    phi = random.uniform(key, (Nphot,), minval=0, maxval=2*jnp.pi)
    theta = jnp.radians(cone_opening) * random.uniform(subkey, (Nphot,))
    
    x = jnp.sin(theta) * jnp.cos(phi)
    y = jnp.sin(theta) * jnp.sin(phi)
    z = jnp.cos(theta)
    
    ray_vectors = jnp.stack([x, y, z], axis=-1)
    
    rotation_matrix = generate_rotation_matrix(track_direction)
    ray_vectors = jnp.einsum('ij,kj->ki', rotation_matrix, ray_vectors)
    
    key, subkey = random.split(key)
    ray_origins = track_origin + random.uniform(subkey, (Nphot, 1)) * track_direction
    
    return ray_vectors, ray_origins

@jax.jit
def generate_rotation_matrix(vector):
    v = normalize(vector)
    z = jnp.array([0., 0., 1.])
    axis = jnp.cross(z, v)
    cos_theta = jnp.dot(z, v)
    sin_theta = jnp.linalg.norm(axis)
    
    # Instead of using an if statement, use jnp.where
    identity = jnp.eye(3)
    flipped = jnp.diag(jnp.array([1., 1., -1.]))
    
    axis = jnp.where(sin_theta > 1e-6, axis / sin_theta, axis)
    K = jnp.array([[0, -axis[2], axis[1]],
                   [axis[2], 0, -axis[0]],
                   [-axis[1], axis[0], 0]])
    
    rotation = identity + sin_theta * K + (1 - cos_theta) * jnp.dot(K, K)
    
    return jnp.where(sin_theta > 1e-6, 
                     rotation, 
                     jnp.where(cos_theta > 0, identity, flipped))


@partial(jax.jit, static_argnums=(5,))
def differentiable_toy_mc_simulator(cone_opening, track_origin, track_direction, detector_points, detector_radius, Nphot, key):
    ray_vectors, ray_origins = differentiable_get_rays(track_origin, track_direction, cone_opening, Nphot, key)

    # Reshape ray_origins and ray_vectors
    ray_origins = ray_origins[:, None, None, :]  # Shape: (Nphot, 1, 1, 3)
    ray_vectors = ray_vectors[:, None, None, :]  # Shape: (Nphot, 1, 1, 3)

    # Create a line parameter array
    t = jnp.linspace(0, 10, 100)[None, :, None, None]  # Shape: (1, 100, 1, 1)

    # Compute points along the rays
    points_along_rays = ray_origins + t * ray_vectors  # Shape: (Nphot, 100, 1, 3)

    # Reshape detector_points
    detector_points = detector_points[None, None, :, :]  # Shape: (1, 1, n_detectors, 3)

    # Compute distances
    distances = jnp.linalg.norm(points_along_rays - detector_points, axis=-1)  # Shape: (Nphot, 100, n_detectors)
    
    min_distances = jnp.min(distances, axis=1)  # Shape: (Nphot, n_detectors)
    hit_probs = jax.nn.sigmoid(-(min_distances - detector_radius) / 0.01)
    
    simulated_histogram = jnp.sum(hit_probs, axis=0)
    
    return simulated_histogram


@partial(jax.jit, static_argnums=(6,))
def smooth_loss_function(true_indices, cone_opening, track_origin, track_direction, detector_points, detector_radius, Nphot, key):
    simulated_histogram = differentiable_toy_mc_simulator(cone_opening, track_origin, track_direction, detector_points, detector_radius, Nphot, key)
    
    true_histogram = jnp.zeros(len(detector_points))
    true_histogram = true_histogram.at[true_indices].add(1)
    
    return jnp.mean((simulated_histogram - true_histogram)**2)


def loss_function(true_indices, cone_opening, track_origin, track_direction, detector):
    simulated_histogram, simulated_indices = toy_mc_simulator(true_indices, cone_opening, track_origin, track_direction, detector)
    
    # Create true histogram
    true_histogram = jnp.zeros(len(detector.all_points), dtype=jnp.int32)
    true_counts = jnp.bincount(true_indices, length=len(detector.all_points))
    true_histogram = true_histogram.at[true_indices].set(true_counts[true_indices])
    
    # Combine true and simulated indices
    all_indices = jnp.unique(jnp.concatenate([true_indices, simulated_indices]))
    
    # Compare only at hit positions
    true_hits = true_histogram[all_indices]
    simulated_hits = simulated_histogram[all_indices]
    
    print("Shape of true_hits:", true_hits.shape)
    print("Shape of simulated_hits:", simulated_hits.shape)

    print(true_hits[0:100])
    print(simulated_hits[0:100])
    
    return jnp.mean((simulated_hits - true_hits)**2)

def main():
    # Set default values
    default_json_filename = 'cyl_geom_config.json'
    output_filename = 'autodiff_datasets/data_events.h5'
    
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description='Description of your program')
    parser.add_argument('--is_data', type=bool, default=False, help='This creates the data event.')
    parser.add_argument('--json_filename', type=str, default=default_json_filename, help='The JSON filename')

    args = parser.parse_args()
    
    json_filename = args.json_filename

    if args.is_data:
        print('Using data mode')
        # Use specific parameters for data generation
        true_cone_opening = 40.
        true_track_origin = np.array([0., 0., 0.])
        true_track_direction = np.array([1., 0., 0.])
        generate_data(json_filename, output_filename, true_cone_opening, true_track_origin, true_track_direction)
    else:
        print('Inference mode')

        detector = generate_detector(json_filename)
        true_indices, _, _, true_cone_opening, true_track_origin, true_track_direction = load_data(output_filename)
        
        # Start with random parameters for inference
        cone_opening = np.random.uniform(20., 60.)
        track_origin = np.random.uniform(-0.4, 0.4, size=3)
        track_direction = normalize(np.random.uniform(-1., 1., size=3))

        key = random.PRNGKey(0)

        Nphot = 100  # or whatever value you want to use
        detector_points = jnp.array(detector.all_points)
        detector_radius = detector.S_radius

        loss_and_grad = jax.value_and_grad(smooth_loss_function, argnums=(1, 2, 3))

        # Optimization parameters
        learning_rate = 0.01
        num_iterations = 100

        print("Initial parameters:")
        print("Cone opening:", cone_opening)
        print("Track origin:", track_origin)
        print("Track direction:", track_direction)

        print("\nTrue parameters:")
        print("Cone opening:", true_cone_opening)
        print("Track origin:", true_track_origin)
        print("Track direction:", true_track_direction)

        # Optimization loop
        for i in range(num_iterations):
            loss, (grad_cone, grad_origin, grad_direction) = loss_and_grad(
                true_indices, cone_opening, track_origin, track_direction, 
                detector_points, detector_radius, Nphot, key
            )
            
            # Update parameters
            cone_opening = cone_opening - learning_rate * grad_cone
            track_origin = track_origin - learning_rate * grad_origin
            track_direction = normalize(track_direction - learning_rate * grad_direction)
            
            # Print progress every 10 iterations
            if i % 10 == 0:
                print(f"Iteration {i}, Loss: {loss}")
                print(f"Cone opening: {cone_opening}")
                print(f"Track origin: {track_origin}")
                print(f"Track direction: {track_direction}")
                print()

        print("\nOptimization complete.")
        print("Final parameters:")
        print(f"Cone opening: {cone_opening}")
        print(f"Track origin: {track_origin}")
        print(f"Track direction: {track_direction}")

        print("\nTrue parameters:")
        print(f"Cone opening: {true_cone_opening}")
        print(f"Track origin: {true_track_origin}")
        print(f"Track direction: {true_track_direction}")

        print(f"\nFinal Loss: {loss}")

if __name__ == "__main__":
    stime = time.perf_counter()
    main()
    print('Total exec. time: ', f"{time.perf_counter()-stime:.2f} s.")
