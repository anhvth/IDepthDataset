"""
    This script visualize rgbd images and depth images.
"""

import open3d as o3d
import argparse
import os
from idd.r3d import pcd_from_r3d

if __name__ == '__main__':
    os.makedirs(".cache", exist_ok=True)
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--input_path', type=str, help='path to the input, could be .depth or .jpg or .conf')
    parser.add_argument('--max_depth', type=float, default=None, help='max depth')
    args = parser.parse_args()

    # Read images
    pcd = pcd_from_r3d(args.input_path)
    o3d.visualization.draw_geometries([pcd])