import open3d as o3d
import argparse
import os
from idd.r3d import pair_rgb_depth_from_r3d


if __name__ == '__main__':
    from avcv.all import *
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_dir')
    parser.add_argument('--output_dir', default=None)
    parser.add_argument('--ext', default='jpg')
    args = parser.parse_args()
    if args.output_dir is None:
        dir_name = os.path.normpath(args.input_dir).split(os.sep)[-2]
        args.output_dir = osp.join('training/datasets/', dir_name)
        logger.info(f'output_dir is not specified, using {args.output_dir}')
    rgb_paths = glob(osp.join(args.input_dir, f'*.{args.ext}'))
    
    def f(rgb_path):
        rgb, depth, conf_img, intrinsics = pair_rgb_depth_from_r3d(rgb_path)

        # Depth are in meters, convert to mm
        
        depth = (depth * 1000).astype(np.uint16)
        valid_mask = np.logical_and(depth > 0, depth < 1000)
        valid_mask = np.logical_and(valid_mask, conf_img == 2)

        depth[~valid_mask] = 0

        out_rgb_path = osp.join(args.output_dir, 'images', osp.basename(rgb_path))
        out_depth_path = osp.join(args.output_dir, 'depths',osp.splitext(osp.basename(rgb_path))[0] + '.png')
        out_segmenation_path = osp.join(args.output_dir, 'segmentations',osp.splitext(osp.basename(rgb_path))[0] + '.png')
        # Make dir if not exist
        os.makedirs(osp.dirname(out_rgb_path), exist_ok=True)
        os.makedirs(osp.dirname(out_depth_path), exist_ok=True)
        os.makedirs(osp.dirname(out_segmenation_path), exist_ok=True)
        bgr = rgb[..., ::-1]
        cv2.imwrite(out_rgb_path, bgr)
        cv2.imwrite(out_depth_path, depth)
        # Make conf_img to segmentation
        conf_img = conf_img.astype(int)
        cv2.imwrite(out_segmenation_path, conf_img)

    multi_thread(f, rgb_paths)
    intrinsics = pair_rgb_depth_from_r3d(rgb_paths[-1])[-1]
    mmcv.dump(intrinsics, osp.join(args.output_dir, 'intrinsics.json'))