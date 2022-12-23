"""
    This script visualize rgbd images and depth images.
"""
import cv2
import argparse
import numpy as np
import os, os.path as osp
from idd.rgbd import visualize_rgbd
import json


if __name__ == '__main__':
    os.makedirs(".cache", exist_ok=True)
    parser = argparse.ArgumentParser()
    parser.add_argument('--rgb', type=str,
                        default='data/rgbd/rgb/000000.png', help='rgb image')
    parser.add_argument(
        '--depth', type=str, default=None, help='depth image')
    parser.add_argument(
        '--conf', type=str, default=None, help='depth image')

    parser.add_argument('--intrinsic', default=None, type=str, help='intrinsic file')
    parser.add_argument('--input_as_dir', action='store_true', help='input as dir')
    parser.add_argument('--max_depth', type=float, default=None, help='max depth')
    parser.add_argument('--pred', action='store_true', help='Use predict instead of ground truth')

    args = parser.parse_args()

    if args.intrinsic is None:
        args.intrinsic = osp.dirname(args.rgb).replace('images', 'intrinsics.json')

    # if args.depth is None:
    if args.pred:
        file_name = args.rgb.split('/')[-1].replace('.jpg', '.png')
        video_name = args.rgb.split('/')[-3]
        args.depth = f'output/{video_name}/depth/{file_name}'
    else:
        args.depth = args.rgb.replace('images', 'depths').replace('.jpg', '.png')

    if args.conf is None:
        args.conf = args.rgb.replace('images', 'segmentations').replace('.jpg', '.png')
        # import ipdb; ipdb.set_trace()
        # args.conf = f'output/{video_name}/conf/{file_name}'

    assert osp.exists(args.depth), args.depth

    if args.intrinsic is not None:
        data = json.load(open(args.intrinsic))
        args.fx, args.fy, args.cx, args.cy = data['fx'], data['fy'], data['cx'], data['cy']
    
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