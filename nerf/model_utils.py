
import functools
from typing import Any, Callable

from flax import linen as nn
import jax
from jax import lax
from jax import random
import jax.numpy as jnp

import numpy as np
import matplotlib.colors as mcolors
from jax import grad, vmap
import cv2



class MLP(nn.Module):
  """A simple MLP."""
  net_depth: int = 8  # The depth of the first part of MLP.
  net_width: int = 256  # The width of the first part of MLP.
  net_activation: Callable[Ellipsis, Any] = nn.relu  # The activation function.
  skip_layer: int = 4  # The layer to add skip layers to.
  num_rgb_channels: int = 3  # The number of RGB channels.
  num_sigma_channels: int = 1  # The number of sigma channels.

  @nn.compact
  def __call__(self, x):
    """Evaluate the MLP.

    Args:
      x: jnp.ndarray(float32), [batch, num_samples, feature], points.

    Returns:
      raw_rgb: jnp.ndarray(float32), with a shape of
           [batch, num_samples, num_rgb_channels].
      raw_sigma: jnp.ndarray(float32), with a shape of
           [batch, num_samples, num_sigma_channels].
    """
    feature_dim = x.shape[-1]
    num_samples = x.shape[1]
    x = x.reshape([-1, feature_dim])
    dense_layer = functools.partial(
        nn.Dense, kernel_init=jax.nn.initializers.glorot_uniform())
    inputs = x
    for i in range(self.net_depth):
      x = dense_layer(self.net_width)(x)
      x = self.net_activation(x)
      if i % self.skip_layer == 0 and i > 0:
        x = jnp.concatenate([x, inputs], axis=-1)
    raw_sigma = dense_layer(self.num_sigma_channels)(x).reshape(
        [-1, num_samples, self.num_sigma_channels])
    raw_rgb = dense_layer(self.num_rgb_channels)(x).reshape(
        [-1, num_samples, self.num_rgb_channels])
    return raw_rgb, raw_sigma


class base_mlp(nn.Module):
    
    activation_func: Callable[Ellipsis, Any] = nn.relu 

    @nn.compact
    def __call__(self, x):
        batch_size  = x.shape[0]
        feature_dim = x.shape[-1]
        num_samples = x.shape[1]
        num_views = x.shape[2]
        x = x.reshape([-1, feature_dim])        
        x = nn.Dense(64)(x)
        x = self.activation_func(x)
        x = nn.Dense(32)(x)
        x = self.activation_func(x)
        return x.reshape(
        [batch_size, num_samples,num_views, -1])

class vis_mlp_1(nn.Module):
    
    activation_func: Callable[Ellipsis, Any] = nn.relu 

    @nn.compact
    def __call__(self, x):
        batch_size  = x.shape[0]
        feature_dim = x.shape[-1]
        num_samples = x.shape[1]
        num_views = x.shape[2]
        x = x.reshape([-1, feature_dim])        
        x = nn.Dense(32)(x)
        x = nn.elu(x)
        x = nn.Dense(33)(x)
        x = nn.elu(x)
        return x.reshape(
        [batch_size, num_samples,num_views, -1])


class vis_mlp_2(nn.Module):
    

    @nn.compact
    def __call__(self, x):
        batch_size  = x.shape[0]
        feature_dim = x.shape[-1]
        num_samples = x.shape[1]
        num_views = x.shape[2]
        x = x.reshape([-1, feature_dim])        
        x = nn.Dense(32)(x)
        x = nn.elu(x)
        x = nn.Dense(1)(x)
        x = nn.sigmoid(x)
        return x.reshape(
        [batch_size, num_samples,num_views, -1])

class geom_mlp(nn.Module):
    

    @nn.compact
    def __call__(self, x):
        
        feature_dim = x.shape[-1]
        num_samples = x.shape[1]
        batch_size = x.shape[0]
        x = x.reshape([-1, feature_dim])        
        x = nn.Dense(64)(x)
        x = nn.elu(x)
        x = nn.Dense(16)(x)
        x = nn.elu(x)
        return x.reshape(
        [batch_size, num_samples, -1])


class geom_mlp_output(nn.Module):
    

    @nn.compact
    def __call__(self, x):
        
        feature_dim = x.shape[-1]
        num_samples = x.shape[1]
        batch_size = x.shape[0]
        x = x.reshape([-1, feature_dim])        
        x = nn.Dense(16)(x)
        x = nn.elu(x)
        x = nn.Dense(3)(x)
        x = nn.elu(x)
        return x.reshape(
        [batch_size, num_samples, -1])




def rgb_to_hsv_batch(images):
    """
    Convert a batch of RGB images to HSV color space.
    
    Parameters:
    - images: A numpy array of shape (batch_size, height, width, channels)
              Assumes images are in RGB format and the range [0, 1].
    
    Returns:
    - A numpy array of images in HSV color space with the same shape.
    """
    batch_size, height, width, _ = images.shape
    hsv_images = np.zeros_like(images)
    
    for i in range(batch_size):
        hsv_images[i] = mcolors.rgb_to_hsv(images[i])
    
    return hsv_images



def compute_channel_gradients(channel):
    # Compute the gradient with respect to x and y for a single channel
    grad_x = jnp.gradient(channel, axis=1)
    grad_y = jnp.gradient(channel, axis=0)
    return grad_x, grad_y

def compute_image_gradients_per_channel(image):
    # Apply gradient computation for each channel and concatenate results
    gradients = vmap(compute_channel_gradients, in_axes=2, out_axes=2)(image)
    # Concatenate gradients along the channel dimension to form [height, width, 6]
    gradients = jnp.concatenate((gradients[0], gradients[1]), axis=2)
    return gradients


def compute_image_gradients(images):
    # Vectorize 'compute_image_gradients_per_channel' over the batch dimension
    batch_gradients = vmap(compute_image_gradients_per_channel, in_axes=0, out_axes=0)(images)
    return batch_gradients



class MultiHeadAttention(nn.Module):
    input_dim: int
    num_heads: int
    ff_ratio: int
    dropout_p: float

    @nn.compact
    def __call__(self, x, mask=None, training=True):
        # Assuming x has shape (batch_size, num_samples, input_dim)
        
        deterministic = not training


        attn_mask = mask if mask is not None else None

        # Define multi-head self-attention
        attn_layer = nn.SelfAttention(
            num_heads=self.num_heads, 
            qkv_features=self.input_dim, 
            out_features=self.input_dim, 
            use_bias=False, 
            deterministic=deterministic,
            dropout_rate=self.dropout_p
        )

        # Define feed-forward layers
        ff_dim = self.input_dim * self.ff_ratio
        ff_layer = nn.Sequential([
            nn.Dense(features=ff_dim),
            nn.gelu,
            nn.Dropout(rate=self.dropout_p, deterministic=deterministic),
            nn.Dense(features=self.input_dim),
            nn.Dropout(rate=self.dropout_p, deterministic=deterministic)
        ])

        # Self-attention block
        attn_out = attn_layer(x, mask=attn_mask)  # Pass the mask directly to the attention layer
        x = nn.LayerNorm()(x + attn_out)  # Apply residual connection and normalization

        # Feed-forward block
        ff_out = ff_layer(x)
        x = nn.LayerNorm()(x + ff_out)  # Apply second residual connection and normalization

        return x


 
def cast_rays(z_vals, origins, directions):
  return origins[Ellipsis, None, :] + z_vals[Ellipsis, None] * directions[Ellipsis, None, :]


def sample_along_rays(key, origins, directions, num_samples, near, far,
                      randomized, lindisp):
  """Stratified sampling along the rays.

  Args:
    key: jnp.ndarray, random generator key.
    origins: jnp.ndarray(float32), [batch_size, 3], ray origins.
    directions: jnp.ndarray(float32), [batch_size, 3], ray directions.
    num_samples: int.
    near: float, near clip.
    far: float, far clip.
    randomized: bool, use randomized stratified sampling.
    lindisp: bool, sampling linearly in disparity rather than depth.

  Returns:
    z_vals: jnp.ndarray, [batch_size, num_samples], sampled z values.
    points: jnp.ndarray, [batch_size, num_samples, 3], sampled points.
  """
  batch_size = origins.shape[0]

  t_vals = jnp.linspace(0., 1., num_samples)
  if lindisp:
    z_vals = 1. / (1. / near * (1. - t_vals) + 1. / far * t_vals)
  else:
    z_vals = near * (1. - t_vals) + far * t_vals

  if randomized:
    mids = .5 * (z_vals[Ellipsis, 1:] + z_vals[Ellipsis, :-1])
    upper = jnp.concatenate([mids, z_vals[Ellipsis, -1:]], -1)
    lower = jnp.concatenate([z_vals[Ellipsis, :1], mids], -1)
    t_rand = random.uniform(key, [batch_size, num_samples])
    z_vals = lower + (upper - lower) * t_rand
  else:
    # Broadcast z_vals to make the returned shape consistent.
    z_vals = jnp.broadcast_to(z_vals[None, Ellipsis], [batch_size, num_samples])

  coords = cast_rays(z_vals, origins, directions)
  return z_vals, coords


def posenc(x, min_deg, max_deg, legacy_posenc_order=False):
  """Cat x with a positional encoding of x with scales 2^[min_deg, max_deg-1].

  Instead of computing [sin(x), cos(x)], we use the trig identity
  cos(x) = sin(x + pi/2) and do one vectorized call to sin([x, x+pi/2]).

  Args:
    x: jnp.ndarray, variables to be encoded. Note that x should be in [-pi, pi].
    min_deg: int, the minimum (inclusive) degree of the encoding.
    max_deg: int, the maximum (exclusive) degree of the encoding.
    legacy_posenc_order: bool, keep the same ordering as the original tf code.

  Returns:
    encoded: jnp.ndarray, encoded variables.
  """
  if min_deg == max_deg:
    return x
  scales = jnp.array([2**i for i in range(min_deg, max_deg)])
  if legacy_posenc_order:
    xb = x[Ellipsis, None, :] * scales[:, None]
    four_feat = jnp.reshape(
        jnp.sin(jnp.stack([xb, xb + 0.5 * jnp.pi], -2)),
        list(x.shape[:-1]) + [-1])
  else:
    xb = jnp.reshape((x[Ellipsis, None, :] * scales[:, None]),
                     list(x.shape[:-1]) + [-1])
    four_feat = jnp.sin(jnp.concatenate([xb, xb + 0.5 * jnp.pi], axis=-1))
  return jnp.concatenate([x] + [four_feat], axis=-1)



def custom_posenc(d_hid, n_samples):
    def get_position_angle_vec(position):
        return [position / jnp.power(10000, 2 * (hid_j // 2) / d_hid) for hid_j in range(d_hid)]

    sinusoid_table = jnp.array([get_position_angle_vec(pos_i) for pos_i in range(n_samples)])
    sinusoid_table = sinusoid_table.at[:, 0::2].set(jnp.sin(sinusoid_table[:, 0::2]))  # dim 2i
    sinusoid_table = sinusoid_table.at[:, 1::2].set(jnp.cos(sinusoid_table[:, 1::2]))  # dim 2i+1

    # In JAX, explicit device placement is less common than in PyTorch,
    # and operations are automatically performed on the GPU if one is available.
    # Therefore, the part about moving to a specific GPU device and converting to float
    # is typically not necessary. JAX handles dtype and device placement more seamlessly.
    sinusoid_table = sinusoid_table[None, :, :]  # Unsqueeze operation in JAX for adding a batch dimension

    return sinusoid_table


def diagonalize_features(features):
    """
    Takes an input array of shape (64, 3) and transforms it into an output array of shape (64, 64, 3).
    In the output array, each of the 64 positions along one axis will contain the original values
    at its corresponding position, with all other positions set to 0.

    :param features: Input array of shape (64, 3).
    :return: Output array of shape (64, 64, 3).
    """
    # Number of features and dimensionality
    n_features, dim = features.shape
    
    # Create an identity matrix of shape (64, 64)
    identity = jnp.eye(n_features)
    
    # Broadcast multiply the identity matrix with the features across a new axis
    # to create the diagonalized feature matrix
    diagonalized_features = identity[:, :, None] * features[None, :, :]
    
    return diagonalized_features



def volumetric_rendering(rgb, sigma, z_vals, dirs, white_bkgd):
  """Volumetric Rendering Function.

  Args:
    rgb: jnp.ndarray(float32), color, [batch_size, num_samples, 3]
    sigma: jnp.ndarray(float32), density, [batch_size, num_samples, 1].
    z_vals: jnp.ndarray(float32), [batch_size, num_samples].
    dirs: jnp.ndarray(float32), [batch_size, 3].
    white_bkgd: bool.

  Returns:
    comp_rgb: jnp.ndarray(float32), [batch_size, 3].
    disp: jnp.ndarray(float32), [batch_size].
    acc: jnp.ndarray(float32), [batch_size].
    weights: jnp.ndarray(float32), [batch_size, num_samples]
  """
  eps = 1e-10
  dists = jnp.concatenate([
      z_vals[Ellipsis, 1:] - z_vals[Ellipsis, :-1],
      jnp.broadcast_to(1e10, z_vals[Ellipsis, :1].shape)
  ], -1)
  dists = dists * jnp.linalg.norm(dirs[Ellipsis, None, :], axis=-1)
  # Note that we're quietly turning sigma from [..., 0] to [...].
  alpha = 1.0 - jnp.exp(-sigma[Ellipsis, 0] * dists)
  accum_prod = jnp.concatenate([
      jnp.ones_like(alpha[Ellipsis, :1], alpha.dtype),
      jnp.cumprod(1.0 - alpha[Ellipsis, :-1] + eps, axis=-1)
  ],
                               axis=-1)
  weights = alpha * accum_prod

  comp_rgb = (weights[Ellipsis, None] * rgb).sum(axis=-2)
  depth = (weights * z_vals).sum(axis=-1)
  acc = weights.sum(axis=-1)
  # Equivalent to (but slightly more efficient and stable than):
  #  disp = 1 / max(eps, where(acc > eps, depth / acc, 0))
  inv_eps = 1 / eps
  disp = acc / depth
  disp = jnp.where((disp > 0) & (disp < inv_eps) & (acc > eps), disp, inv_eps)
  if white_bkgd:
    comp_rgb = comp_rgb + (1. - acc[Ellipsis, None])
  return comp_rgb, disp, acc, weights


def piecewise_constant_pdf(key, bins, weights, num_samples, randomized):
  """Piecewise-Constant PDF sampling.

  Args:
    key: jnp.ndarray(float32), [2,], random number generator.
    bins: jnp.ndarray(float32), [batch_size, num_bins + 1].
    weights: jnp.ndarray(float32), [batch_size, num_bins].
    num_samples: int, the number of samples.
    randomized: bool, use randomized samples.

  Returns:
    z_samples: jnp.ndarray(float32), [batch_size, num_samples].
  """
  # Pad each weight vector (only if necessary) to bring its sum to `eps`. This
  # avoids NaNs when the input is zeros or small, but has no effect otherwise.
  eps = 1e-5
  weight_sum = jnp.sum(weights, axis=-1, keepdims=True)
  padding = jnp.maximum(0, eps - weight_sum)
  weights += padding / weights.shape[-1]
  weight_sum += padding

  # Compute the PDF and CDF for each weight vector, while ensuring that the CDF
  # starts with exactly 0 and ends with exactly 1.
  pdf = weights / weight_sum
  cdf = jnp.minimum(1, jnp.cumsum(pdf[Ellipsis, :-1], axis=-1))
  cdf = jnp.concatenate([
      jnp.zeros(list(cdf.shape[:-1]) + [1]), cdf,
      jnp.ones(list(cdf.shape[:-1]) + [1])
  ],
                        axis=-1)

  # Draw uniform samples.
  if randomized:
    # Note that `u` is in [0, 1) --- it can be zero, but it can never be 1.
    u = random.uniform(key, list(cdf.shape[:-1]) + [num_samples])
  else:
    # Match the behavior of random.uniform() by spanning [0, 1-eps].
    u = jnp.linspace(0., 1. - jnp.finfo('float32').eps, num_samples)
    u = jnp.broadcast_to(u, list(cdf.shape[:-1]) + [num_samples])

  # Identify the location in `cdf` that corresponds to a random sample.
  # The final `True` index in `mask` will be the start of the sampled interval.
  mask = u[Ellipsis, None, :] >= cdf[Ellipsis, :, None]

  def find_interval(x):
    # Grab the value where `mask` switches from True to False, and vice versa.
    # This approach takes advantage of the fact that `x` is sorted.
    x0 = jnp.max(jnp.where(mask, x[Ellipsis, None], x[Ellipsis, :1, None]), -2)
    x1 = jnp.min(jnp.where(~mask, x[Ellipsis, None], x[Ellipsis, -1:, None]), -2)
    return x0, x1

  bins_g0, bins_g1 = find_interval(bins)
  cdf_g0, cdf_g1 = find_interval(cdf)

  t = jnp.clip(jnp.nan_to_num((u - cdf_g0) / (cdf_g1 - cdf_g0), 0), 0, 1)
  samples = bins_g0 + t * (bins_g1 - bins_g0)

  # Prevent gradient from backprop-ing through `samples`.
  return lax.stop_gradient(samples)


def sample_pdf(key, bins, weights, origins, directions, z_vals, num_samples,
               randomized):
  """Hierarchical sampling.

  Args:
    key: jnp.ndarray(float32), [2,], random number generator.
    bins: jnp.ndarray(float32), [batch_size, num_bins + 1].
    weights: jnp.ndarray(float32), [batch_size, num_bins].
    origins: jnp.ndarray(float32), [batch_size, 3], ray origins.
    directions: jnp.ndarray(float32), [batch_size, 3], ray directions.
    z_vals: jnp.ndarray(float32), [batch_size, num_coarse_samples].
    num_samples: int, the number of samples.
    randomized: bool, use randomized samples.

  Returns:
    z_vals: jnp.ndarray(float32),
      [batch_size, num_coarse_samples + num_fine_samples].
    points: jnp.ndarray(float32),
      [batch_size, num_coarse_samples + num_fine_samples, 3].
  """
  z_samples = piecewise_constant_pdf(key, bins, weights, num_samples,
                                     randomized)
  # Compute united z_vals and sample points
  z_vals = jnp.sort(jnp.concatenate([z_vals, z_samples], axis=-1), axis=-1)
  coords = cast_rays(z_vals, origins, directions)
  return z_vals, coords


def add_gaussian_noise(key, raw, noise_std, randomized):
  """Adds gaussian noise to `raw`, which can used to regularize it.

  Args:
    key: jnp.ndarray(float32), [2,], random number generator.
    raw: jnp.ndarray(float32), arbitrary shape.
    noise_std: float, The standard deviation of the noise to be added.
    randomized: bool, add noise if randomized is True.

  Returns:
    raw + noise: jnp.ndarray(float32), with the same shape as `raw`.
  """
  if (noise_std is not None) and randomized:
    return raw + random.normal(key, raw.shape, dtype=raw.dtype) * noise_std
  else:
    return raw



def map_pixels_to_features(pixel_coords, features ):
    # Ensure the inputs are JAX arrays (if they're not already)
    features = jnp.array(features)
    pixel_coords = jnp.array(pixel_coords)
    
    # Extract x and y coordinates
    # Note: pixel_coords[..., 0] represents x, pixel_coords[..., 1] represents y
    x_coords = pixel_coords[..., 0]
    y_coords = pixel_coords[..., 1]
    
    # Ensure coordinates are within bounds, assuming features are indexed as [height, width]
    x_coords = jnp.clip(x_coords, 0, features.shape[1] - 1)
    y_coords = jnp.clip(y_coords, 0, features.shape[2] - 1)
    
    # Convert coordinates to integer for indexing
    x_coords = x_coords.astype(jnp.int32)
    y_coords = y_coords.astype(jnp.int32)
    
    # Gather the corresponding features using advanced indexing
    batch_indices = jnp.arange(features.shape[0])
    # Repeat and tile batch_indices to match the shape of x_coords and y_coords
    batch_indices = jnp.repeat(batch_indices[None, None, :], pixel_coords.shape[0], axis=0)
    batch_indices = jnp.repeat(batch_indices, pixel_coords.shape[1], axis=1)
    
    # Use advanced indexing to gather the features
    gathered_features = features[batch_indices, x_coords, y_coords]
    
    return gathered_features




def find_closest_cameras(target_extrinsic, source_extrinsics):
    # Calculate the positions of target and source cameras
    target_position = target_extrinsic[:3, 3]
    source_positions = source_extrinsics[:, :3, 3]

    # Calculate Euclidean distances between the target and all source cameras
    distances = np.sqrt(np.sum((source_positions - target_position) ** 2, axis=1))

    # Select indices of the 50 nearest source views based on spatial distance
    nearest_50_indices = np.argsort(distances)[:50]

    # Compute directions of the cameras (assuming the forward direction is the third column of the rotation matrix)
    target_direction = target_extrinsic[0:3, 2]  # Using the forward vector (z-axis)
    source_directions = source_extrinsics[nearest_50_indices, 0:3, 2]

    # Normalize the vectors to compute the dot product correctly
    target_direction_norm = target_direction / np.linalg.norm(target_direction)
    source_directions_norm = source_directions / np.linalg.norm(source_directions, axis=1)[:, None]

    # Calculate the dot product for cosine similarity
    cos_angles = np.dot(source_directions_norm, target_direction_norm)

    # Since cosine similarity might be negative, we sort in descending order to get the closest in terms of angle
    closest_25_indices_from_50 = np.argsort(-cos_angles)[:25]
    closest_25_indices = nearest_50_indices[closest_25_indices_from_50]

    # Return the extrinsics and indices of the 25 closest views
    closest_25_extrinsics = source_extrinsics[closest_25_indices]
    return closest_25_extrinsics, closest_25_indices




def compute_projections(xyz, train_cameras,intrinsics):
    '''
    project 3D points into cameras
    :param xyz: [..., 3]
    :param train_cameras: [souceviews,4,4]
    :param intrinsics: [souceviews,4,4]
    :return: pixel locations [..., 2], mask [...]
    '''
    original_shape = xyz.shape[:2]
    xyz = xyz.reshape(-1, 3)
    num_views = train_cameras.shape[0]
    xyz_h = jnp.concatenate([xyz, jnp.ones_like(xyz[..., :1])], axis=-1) 

    # Compute the inverse of train_poses
    train_poses_inv = jnp.linalg.inv(train_cameras)

    # Batch matrix multiplication of train_intrinsics and the inverse of train_poses
    projections = jnp.matmul(intrinsics, train_poses_inv)

    # Transpose xyz_h and add a new axis to match batch dimension
    xyz_h_t = xyz_h.T[None, ...]

    # Repeat xyz_h_t across the batch dimension (num_views times)
    xyz_h_t_repeat = jnp.repeat(xyz_h_t, num_views, axis=0)

    # Batch matrix multiplication of projections with the repeated xyz_h tensor
    projections = jnp.matmul(projections, xyz_h_t_repeat)

    projections = jnp.transpose(projections, axes=(0, 2, 1))  # [n_views, n_points, 4]
    pixel_locations = projections[..., :2] / jnp.clip(projections[..., 2:3], a_min=1e-8)  # [n_views, n_points, 2]
    pixel_locations = jnp.clip(pixel_locations, a_min=-1.e6, a_max=1.e6)
    mask = projections[..., 2] > 0   # a point is invalid if behind the camera
    return pixel_locations.reshape((num_views, ) + original_shape + (2, )), \
            mask.reshape((num_views, ) + original_shape)


def normalize( pixel_locations, h, w):
    resize_factor = jnp.asarray([w-1.,h-1.])[None, None, :]
    print(resize_factor.shape)
    normalized_pixel_locations = 2 * pixel_locations / resize_factor - 1.  # [n_views, n_points, 2]
    return normalized_pixel_locations

def inbound(pixel_locations, h, w):
    '''
    check if the pixel locations are in valid range
    :param pixel_locations: [..., 2]
    :param h: height
    :param w: weight
    :return: mask, bool, [...]
    '''
    return (pixel_locations[..., 0] <= w - 1.) & \
            (pixel_locations[..., 0] >= 0) & \
            (pixel_locations[..., 1] <= h - 1.) &\
            (pixel_locations[..., 1] >= 0)


def compute_angle(xyz, query_camera, train_cameras):
    '''
    :param xyz: [..., 3]
    :param query_camera: [34, ]
    :param train_cameras: [n_views, 34]
    :return: [n_views, ..., 4]; The first 3 channels are unit-length vector of the difference between
    query and target ray directions, the last channel is the inner product of the two directions.
    '''
    original_shape = xyz.shape[:-1]
    xyz = xyz.reshape(-1, 3)
    num_views = train_cameras.shape[0]
    query_pose = query_camera.reshape(-1, 4, 4)  # Adjusted to reflect input shape
    ray2tar_pose = (query_pose[:, :3, 3][:, jnp.newaxis, :] - xyz[jnp.newaxis, :, :])
    ray2tar_pose /= (jnp.linalg.norm(ray2tar_pose, axis=-1, keepdims=True) + 1e-6)
    
    # Assuming train_poses is similar to query_pose and has shape [n_views, 4, 4]
    ray2train_pose = (train_cameras[:, :3, 3][:, jnp.newaxis, :] - xyz[jnp.newaxis, :, :])
    ray2train_pose /= (jnp.linalg.norm(ray2train_pose, axis=-1, keepdims=True) + 1e-6)
    
    ray_diff = ray2tar_pose - ray2train_pose
    ray_diff_norm = jnp.linalg.norm(ray_diff, axis=-1, keepdims=True)
    ray_diff_dot = jnp.sum(ray2tar_pose * ray2train_pose, axis=-1, keepdims=True)
    ray_diff_direction = ray_diff / jnp.clip(ray_diff_norm, a_min=1e-6)
    ray_diff = jnp.concatenate([ray_diff_direction, ray_diff_dot], axis=-1)
    ray_diff = ray_diff.reshape((num_views,) + original_shape + (4,))
    return ray_diff


def fused_mean_variance(x, weight):
    mean = jnp.sum(x * weight, axis=2, keepdims=True)
    var = jnp.sum(weight * (x - mean) ** 2, axis=2, keepdims=True)
    return mean, var


from typing import Optional

def linear_to_srgb(linear: jnp.ndarray, eps: Optional[float] = None) -> jnp.ndarray:
    """Converts linear RGB values to sRGB using JAX. Assumes `linear` is in [0, 1].
    
    Args:
        linear: Linear RGB values.
        eps: A small epsilon value to avoid taking the root of zero.
    
    Returns:
        Converted sRGB values.
    """
    if eps is None:
        eps = jnp.finfo(jnp.float32).eps
    srgb0 = 323 / 25 * linear
    srgb1 = (211 * jnp.maximum(eps, linear) ** (5 / 12) - 11) / 200
    return jnp.where(linear <= 0.0031308, srgb0, srgb1)



def grid_sample(inp, grid):
    N, H, W, C = inp.shape
    _, H_out, W_out, _ = grid.shape

    # Convert normalized coordinates (-1, 1) to absolute coordinates
    grid = (grid + 1) * jnp.array([[H - 1, W - 1]]) / 2

    i, j = grid[..., 0], grid[..., 1]
    i = jnp.clip(i, 0, H - 1)
    j = jnp.clip(j, 0, W - 1)

    i_1 = jnp.floor(i).astype(jnp.int32)
    j_1 = jnp.floor(j).astype(jnp.int32)
    i_2 = jnp.clip(i_1 + 1, 0, H - 1)
    j_2 = jnp.clip(j_1 + 1, 0, W - 1)

    # Gather pixel values for four corners
    batch_indices = jnp.arange(N)[:, None, None]
    q_11 = inp[batch_indices, i_1, j_1]
    q_12 = inp[batch_indices, i_1, j_2]
    q_21 = inp[batch_indices, i_2, j_1]
    q_22 = inp[batch_indices, i_2, j_2]

    # Compute interpolation weights
    di = i - i_1.astype(i.dtype)
    dj = j - j_1.astype(j.dtype)
    di = di[..., None]
    dj = dj[..., None]

    # Perform bilinear interpolation
    q_i1 = q_11 * (1 - di) + q_21 * di
    q_i2 = q_12 * (1 - di) + q_22 * di
    q_ij = q_i1 * (1 - dj) + q_i2 * dj

    return q_ij


def map_pixels_to_features(pixel_coords, features ):
    # Ensure the inputs are JAX arrays (if they're not already)
    features = jnp.array(features)
    pixel_coords = jnp.array(pixel_coords)
    
    # Extract x and y coordinates
    # Note: pixel_coords[..., 0] represents x, pixel_coords[..., 1] represents y
    x_coords = pixel_coords[..., 0]
    y_coords = pixel_coords[..., 1]
    
    # Ensure coordinates are within bounds, assuming features are indexed as [height, width]
    x_coords = jnp.clip(x_coords, 0, features.shape[1] - 1)
    y_coords = jnp.clip(y_coords, 0, features.shape[2] - 1)
    
    # Convert coordinates to integer for indexing
    x_coords = x_coords.astype(jnp.int32)
    y_coords = y_coords.astype(jnp.int32)
    
    # Gather the corresponding features using advanced indexing
    batch_indices = jnp.arange(features.shape[0])
    # Repeat and tile batch_indices to match the shape of x_coords and y_coords
    batch_indices = jnp.repeat(batch_indices[None, None, :], pixel_coords.shape[0], axis=0)
    batch_indices = jnp.repeat(batch_indices, pixel_coords.shape[1], axis=1)
    
    # Use advanced indexing to gather the features
    gathered_features = features[batch_indices, x_coords, y_coords]
    
    return gathered_features