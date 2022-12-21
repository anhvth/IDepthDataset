"""
    This script visualize rgbd images and depth images.
"""
import cv2
import argparse
import numpy as np
import os
from idd.rgbd import visualize_rgbd



if __name__ == '__main__':
    os.makedirs(".cache", exist_ok=True)
    parser = argparse.ArgumentParser()
    parser.add_argument('--rgb', type=str,
                        default='data/rgbd/rgb/000000.png', help='rgb image')
    parser.add_argument(
        '--depth', type=str, default=None, help='depth image')
    parser.add_argument(
        '--conf', type=str, default=None, help='depth image')
    parser.add_argument('--fx', type=float, default=608.17, help='fx')
    parser.add_argument('--fy', type=float, default=608.17, help='fy')
    parser.add_argument('--cx', type=float, default=327.45, help='cx')
    parser.add_argument('--cy', type=float, default=238.49, help='cy')
    parser.add_argument('--intrinsic', default=None, type=str, help='intrinsic file')
    parser.add_argument('--input_as_dir', action='store_true', help='input as dir')

    parser.add_argument('--max_depth', type=float, default=None, help='max depth')
    args = parser.parse_args()
    if args.intrinsic is not None:
        import json
        data = json.load(open(args.intrinsic))
        print('Load intrinsic from {}'.format(args.intrinsic))
        args.fx, args.fy, args.cx, args.cy = data['fx'], data['fy'], data['cx'], data['cy']
    if args.depth is None:
        args.depth = args.rgb.replace('images', 'depths').replace('.jpg', '.png')
    # Read images
    if args.input_as_dir:
        from glob import glob
        depth_dir = os.path.dirname(args.depth)
        path_depth = glob(f'{depth_dir}/*.png')
        img_dir = os.path.dirname(args.rgb)
        path_depth.sort()
        for i, depth_path in enumerate(path_depth):
            depth = cv2.imread(depth_path, cv2.IMREAD_ANYDEPTH)
            rgb_path = os.path.join(img_dir, os.path.basename(depth_path).replace('.png', '.jpg'))
            output_path = os.path.join('.cache', os.path.basename(depth_path).replace('.png', '.ply'))
            visualize_rgbd(depth, args.fx, args.fy, args.cx, args.cy, rgb_path, args.max_depth, out_file=output_path)
            print('Processing {}/{}'.format(i, len(path_depth)), '->', output_path)

    else:
        depth = cv2.imread(args.depth, cv2.IMREAD_ANYDEPTH)
        visualize_rgbd(depth, args.fx, args.fy, args.cx, args.cy, rgb=args.rgb, conf=args.conf, max_depth=args.max_depth)