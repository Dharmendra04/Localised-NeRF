
INTERNAL = False  # pylint: disable=g-statement-before-imports
import json
import os
from os import path
import queue
import threading
if not INTERNAL:
  import cv2  # pylint: disable=g-import-not-at-top
import jax
import numpy as np
from PIL import Image
from localised_nerf.nerf import utils
from localised_nerf.nerf import model_utils
import imageio
import copy


def get_dataset(split, args):
  return dataset_dict[args.dataset](split, args)


def convert_to_ndc(origins, directions, focal, w, h, near=1.):
  """Convert a set of rays to NDC coordinates."""
  # Shift ray origins to near plane
  t = -(near + origins[Ellipsis, 2]) / directions[Ellipsis, 2]
  origins = origins + t[Ellipsis, None] * directions

  dx, dy, dz = tuple(np.moveaxis(directions, -1, 0))
  ox, oy, oz = tuple(np.moveaxis(origins, -1, 0))

  # Projection
  o0 = -((2 * focal) / w) * (ox / oz)
  o1 = -((2 * focal) / h) * (oy / oz)
  o2 = 1 + 2 * near / oz

  d0 = -((2 * focal) / w) * (dx / dz - ox / oz)
  d1 = -((2 * focal) / h) * (dy / dz - oy / oz)
  d2 = -2 * near / oz

  origins = np.stack([o0, o1, o2], -1)
  directions = np.stack([d0, d1, d2], -1)
  return origins, directions



def get_rays_batch(H, W, intrinsics_batch, c2w_batch):
    """
    Create rays for a batch of images using NumPy.
    :param H: image height
    :param W: image width
    :param intrinsics_batch: batch of camera intrinsics matrices, assumed to be [N, 4, 4]
    :param c2w_batch: batch of camera to world transform matrices, assumed to be [N, 4, 4]
    :return: origins_batch, directions_batch, both reshaped to [N, H, W, 3]
    """
    N = intrinsics_batch.shape[0]  # Batch size
    u, v = np.meshgrid(
        np.arange(W, dtype=np.float32),
        np.arange(H, dtype=np.float32),
        indexing="xy"
    )
    u, v = u.reshape(-1), v.reshape(-1)
    ones = np.ones_like(u)
    pixels = np.stack((u, v, ones), axis=-1)  # (H*W, 3)

    origins_batch = np.empty((N, H * W, 3), dtype=np.float32)
    directions_batch = np.empty((N, H * W, 3), dtype=np.float32)

    for i in range(N):
        intrinsics = intrinsics_batch[i]
        c2w = c2w_batch[i]

        inv_intrinsics = np.linalg.inv(intrinsics[:3, :3])
        dirs = pixels @ inv_intrinsics.T

        rays_d = dirs @ c2w[:3, :3].T
        rays_o = np.broadcast_to(c2w[:3, 3], rays_d.shape)

        origins_batch[i] = rays_o
        directions_batch[i] = rays_d

    # Reshape the outputs to [N, H, W, 3]
    origins_batch = origins_batch.reshape(N, H, W, 3)
    directions_batch = directions_batch.reshape(N, H, W, 3)

    viewdirs_batch = directions_batch / np.linalg.norm(directions_batch, axis=-1, keepdims=True)

    return origins_batch, directions_batch,viewdirs_batch


class Dataset(threading.Thread):
  """Dataset Base Class."""

  def __init__(self, split, args):
    super(Dataset, self).__init__()
    self.queue = queue.Queue(3)  # Set prefetch buffer to 3 batches.
    self.daemon = True
    self.split = split
    if split == "train":
      self._train_init(args)
      self.n = 0
    elif split == "test":
      self._test_init(args)
    else:
      raise ValueError(
          "the split argument should be either \"train\" or \"test\", set"
          "to {} here.".format(split))
    self.batch_size = args.batch_size // jax.host_count()
    self.batching = args.batching
    self.start()

  def __iter__(self):
    return self

  def __next__(self):
    """Get the next training batch or test example.

    Returns:
      batch: dict, has "pixels" and "rays".
    """
    x = self.queue.get()
    if self.split == "train":
      return utils.shard(x)
    else:
      return utils.to_device(x)

  def peek(self):
    """Peek at the next training batch or test example without dequeuing it.

    Returns:
      batch: dict, has "pixels" and "rays".
    """
    x = self.queue.queue[0].copy()  # Make a copy of the front of the queue.
    if self.split == "train":
      return utils.shard(x)
    else:
      return utils.to_device(x)

  def run(self):
    if self.split == "train":
      next_func = self._next_train
    else:
      next_func = self._next_test
    while True:
      self.queue.put(next_func())

  @property
  def size(self):
    return self.n_examples

  def _train_init(self, args):
    """Initialize training."""
    self._load_renderings(args)
    self._generate_rays()


    if args.batching == "single_image":
      self.images = self.images.reshape([-1, self.resolution, 3])
      #features = features.reshape([-1, self.resolution, 1])
      # self.hsv = hsv.reshape([-1, self.w,self.h, 3])
      # self.intensity = intensity.reshape([-1, self.w,self.h, 3])

      self.rays = utils.namedtuple_map(
          lambda r: r.reshape([-1, self.resolution, r.shape[-1]]), self.rays)

    else:
      raise NotImplementedError(
          f"{args.batching} batching strategy is not implemented.")

  def _test_init(self, args):
    self._load_renderings(args)
    self._generate_rays()
    self.it = 0

  def _next_train(self):
    """Sample next training batch."""

    if self.batching == "single_image":

      training_image_index = np.random.randint(0, self.n_examples, ())

      ray_indices = np.random.randint(0, self.rays[0][0].shape[0],
                                      (self.batch_size,))  #self.rays[0][0].shape[0] - 160000

      
      #ray_indices = np.arange(ray_indices,ray_indices + self.batch_size)
    

      self.image_list  = self.image_list_org.copy()
      self.cam_list  = self.cam_list_org.copy()

      cam2target = self.cam_list[training_image_index]

      self.image_list.pop(training_image_index)
      self.cam_list.pop(training_image_index)



      self.image_list = np.stack(self.image_list,axis=0)
      self.cam_list = np.stack(self.cam_list,axis=0)

      

      self.cam_list,index = model_utils.find_closest_cameras(cam2target, self.cam_list)
      self.image_list = self.image_list[index]


      hsv = model_utils.rgb_to_hsv_batch(self.image_list)
      intensity = model_utils.compute_image_gradients(self.image_list)
      intensity = np.concatenate([intensity,self.image_list],axis=-1)
      



      batch_pixels = self.images[training_image_index][ray_indices]
      

      batch_rays = utils.namedtuple_map(lambda r: r[training_image_index][ray_indices],
                                        self.rays)
      
      
      supplement = utils.supplement(hsv = hsv, intensity = intensity,cam2world = self.cam_list,intrinsics = self.intrinsics,focal = self.focal,cam2target = cam2target)
      

    else:
      raise NotImplementedError(
          f"{self.batching} batching strategy is not implemented.")
    return {"pixels": batch_pixels, "rays": batch_rays,'supplement': supplement}
   

  def _next_test(self):
    """Sample next test example."""
    image_index = self.it
    self.it = (self.it + 1) % self.n_examples

    self.image_list  = self.image_list_org.copy()
    self.cam_list  = self.cam_list_org.copy()

    cam2target = self.cam_list[image_index]

    self.image_list.pop(image_index)
    self.cam_list.pop(image_index)

    focal = np.repeat(self.focal,len(self.image_list),axis=-1)
    self.image_list = np.stack(self.image_list,axis=0)
    self.cam_list = np.stack(self.cam_list,axis=0)


    self.cam_list,index = model_utils.find_closest_cameras(cam2target, self.cam_list)
    self.image_list = self.image_list[index]


    hsv = model_utils.rgb_to_hsv_batch(self.image_list)
    intensity = model_utils.compute_image_gradients(self.image_list)

    intensity = np.concatenate([intensity,self.image_list],axis=-1)



    batch_pixels = self.images[image_index]


    supplement = utils.supplement(hsv = hsv, intensity = intensity,cam2world = self.cam_list,intrinsics = self.intrinsics,focal = focal,cam2target = cam2target)
    
    

    return {
       "pixels": self.images[image_index],
       "rays": utils.namedtuple_map(lambda r: r[image_index], self.rays),
       'supplement': supplement,  
}

  def _generate_rays(self):
    """Generating rays for all images."""
    origins, directions, viewdirs = get_rays_batch(
                                                     self.h, self.w, self.intrinsics,
                                                     self.camtoworlds)
    self.rays = utils.Rays(
        origins=origins, directions=directions, viewdirs=viewdirs)


class Blender(Dataset):
  """Blender Dataset."""

  def _load_renderings(self, args):
    """Load images from disk."""
    with utils.open_file(
        path.join(args.data_dir, "transforms_{}.json".format(self.split)),
        "r") as fp:
      meta = json.load(fp)
    images = []
    cams = []
    for i in range(len(meta["frames"])):
      frame = meta["frames"][i]
      fname = os.path.join(args.data_dir, frame["file_path"] + ".png")
      with utils.open_file(fname, "rb") as imgin:
        image = np.array(Image.open(imgin), dtype=np.float32) / 255.
        if args.factor == 2:
          [halfres_h, halfres_w] = [hw // 2 for hw in image.shape[:2]]
          image = cv2.resize(
              image, (halfres_w, halfres_h), interpolation=cv2.INTER_AREA)
        elif args.factor > 0:
          raise ValueError("Blender dataset only supports factor=0 or 2, {} "
                           "set.".format(args.factor))
      c2w = np.array(frame["transform_matrix"], dtype=np.float32)
      w2c_blender = np.linalg.inv(c2w)
      w2c_opencv = w2c_blender
      w2c_opencv[1:3] *= -1.
      c2w_opencv = np.linalg.inv(w2c_opencv)
      cams.append(c2w_opencv)
      images.append(image)
    
    self.images = np.stack(images, axis=0)
    if args.white_bkgd:
      self.images = (
          self.images[Ellipsis, :3] * self.images[Ellipsis, -1:] +
          (1. - self.images[Ellipsis, -1:]))
    else:
      self.images = self.images[Ellipsis, :3]

    self.image_list_org = [self.images[i] for i in range(self.images.shape[0])]
    self.h, self.w = self.images.shape[1:3]
    
    self.resolution = self.h * self.w
    self.camtoworlds = np.stack(cams, axis=0)
    self.cam_list_org = cams
    camera_angle_x = float(meta["camera_angle_x"])
    self.focal = .5 * self.w / np.tan(.5 * camera_angle_x)
    focal_array = np.full((self.camtoworlds.shape[0], 1),self.focal)


    intrinsic_matrices = np.zeros((self.images.shape[0], 4, 4))
    for i in range(self.images.shape[0]):
      f = focal_array[i][0]
      cx, cy = self.h/2,self.w/2
      intrinsic_matrices[i,:,:] = np.array([
          [f, 0, cx, 0],
          [0, f, cy, 0],
          [0, 0, 1, 0],
          [0, 0, 0, 1]
      ])

    self.intrinsics = intrinsic_matrices
    self.n_examples = self.images.shape[0]



dataset_dict = {
    "blender": Blender,
}


