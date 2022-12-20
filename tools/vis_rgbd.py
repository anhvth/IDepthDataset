"""
    This script visualize rgbd images and depth images.
"""

import cv2
import open3d as o3d
import argparse
import matplotlib.pyplot as plt
import numpy as np
import os

os.makedirs(".cache", exist_ok=True)

def depth_map_to_point_cloud(depth_map, fx, fy, cx, cy):
    rows, cols = depth_map.shape[:2]
    # Create arrays to hold the x, y, and z coordinates
    x = np.empty((rows, cols), dtype=np.float32)
    y = np.empty((rows, cols), dtype=np.float32)
    z = np.empty((rows, cols), dtype=np.float32)
    # Calculate the 3D coordinates for each pixel
    x = (np.arange(cols) - cx) * depth_map / fx
    y = (np.arange(rows)[:, np.newaxis] - cy) * depth_map / fy
    z = depth_map
    point_cloud = np.stack((x, y, z), axis=-1)
    return point_cloud


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--rgb', type=str,
                        default='data/rgbd/rgb/000000.png', help='rgb image')
    parser.add_argument(
        '--depth', type=str, default='data/rgbd/depth/000000.png', help='depth image')
    parser.add_argument('--fx', type=float, default=608.17, help='fx')
    parser.add_argument('--fy', type=float, default=608.17, help='fy')
    parser.add_argument('--cx', type=float, default=327.45, help='cx')
    parser.add_argument('--cy', type=float, default=238.49, help='cy')
    parser.add_argument('--max_depth', type=float, default=None, help='max depth')
    args = parser.parse_args()

    # Read images
    depth = np.load(args.depth)
    

    xyz = depth_map_to_point_cloud(depth, args.fx, args.fy, args.cx, args.cy)
    rgb = cv2.imread(args.rgb)[..., ::-1]

    pcd = o3d.geometry.PointCloud()

    if args.max_depth is not None:
        mask = depth < args.max_depth
        xyz = xyz[mask]
        rgb = rgb[mask]

    pcd.points = o3d.utility.Vector3dVector(xyz.reshape(-1, 3))
    pcd.colors = o3d.utility.Vector3dVector(rgb.reshape(-1, 3) / 255.0)
    o3d.visualization.draw_geometries([pcd])

