import open3d as o3d
import argparse
import os
from idd.r3d import pair_rgb_depth_from_r3d


if __name__ == '__main__':
    from avcv.all import *
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_dir')
    parser.add_argument('--output_dir')
    parser.add_argument('--ext', default='jpg')
    args = parser.parse_args()

    rgb_paths = glob(osp.join(args.input_dir, f'*.{args.ext}'))
    # for rgb_path in tqdm(rgb_paths):
    def f(rgb_path):
        rgb, depth, conf_img, intrinsics = pair_rgb_depth_from_r3d(rgb_path)

        # Depth are in meters, convert to mm
        
        depth = (depth * 1000).astype(np.uint16)
        valid_mask = np.logical_and(depth > 0, depth < 3000)
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
        segmentation = np.zeros_like(conf_img)
        segmentation[conf_img == 2] = 255
        cv2.imwrite(out_segmenation_path, segmentation)

    multi_thread(f, rgb_paths)
    intrinsics = pair_rgb_depth_from_r3d(rgb_paths[-1])[-1]
    mmcv.dump(intrinsics, osp.join(args.output_dir, 'intrinsics.json'))