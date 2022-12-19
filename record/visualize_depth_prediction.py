"""
    This code demonstrates how to open an r3d file and display it with open3d
"""

# install liblzfse with pip install pyliblzfse open3d
from r3d import load_r3d
import open3d as o3d
from avcv.all import *


import cv2
import mediapipe as mp
import numpy as np

mp_drawing = mp.solutions.drawing_utils
mp_selfie_segmentation = mp.solutions.selfie_segmentation

class SelfieSegmentation:
  def __init__(self, model_selection=0, bg_color=(0,0,0), mask_color=(255, 255, 255)):
    self.model_selection = model_selection
    self.bg_color = bg_color
    self.mask_color = mask_color
    
  def segment(self, image):
    with mp_selfie_segmentation.SelfieSegmentation(self.model_selection) as selfie_segmentation:
    #   for idx, file in enumerate(image_files):
        # image = cv2.imread(file)
        image_height, image_width, _ = image.shape
        results = selfie_segmentation.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

        condition = np.stack((results.segmentation_mask,) * 3, axis=-1) > 0.1
        # Generate solid color images for showing the output selfie segmentation mask.
        fg_image = np.zeros(image.shape, dtype=np.uint8)
        fg_image[:] = self.mask_color
        bg_image = np.zeros(image.shape, dtype=np.uint8)
        bg_image[:] = self.bg_color
        output_image = np.where(condition, fg_image, bg_image)
    return output_image


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('input')
    parser.add_argument('depth_dir')
    parser.add_argument('--conf', type=float, default=0.5, help='minimum confidence')
    args = parser.parse_args()

    name = get_name(args.input)
    depth_path = os.path.join(args.depth_dir, name + '.png')
    depth_map = mmcv.imread(depth_path, 'unchanged')
    depth_map = depth_map.astype(np.float32) / 255 * 1.5
    depth_map = np.clip(depth_map, 0, 1.5)

    rgb, xyz, conf, _ = load_r3d(args.input, min_conf=args.conf, pred_depth_map=depth_map, reshape=False)
    segmenter = SelfieSegmentation()
    out = segmenter.segment(rgb)
    xyz[...,-1] = xyz[...,-1] + np.random.uniform(*xyz[...,-1].shape) * 0.02
    # import ipdb; ipdb.set_trace()

    # Only keep FG
    mask = out[:, :, 0] > 0
    xyz = xyz[mask]
    rgb = rgb[mask]

    # Flatten xyz and rgb
    xyz = xyz.reshape(-1, 3)
    rgb = rgb.reshape(-1, 3)
    rgb = rgb.astype(np.float32) / 255
    # import ipdb; ipdb.set_trace()


    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(xyz)
    pcd.colors = o3d.utility.Vector3dVector(rgb)
    o3d.io.write_point_cloud("uni_test.ply", pcd)