import open3d as o3d
import argparse
import os
from idd.r3d import pair_rgb_depth_from_r3d


if __name__ == '__main__':
    from avcv.all import *
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_dir', '-i')
    parser.add_argument('--output_dir', '-o')
    args = parser.parse_args()

    

    
    def f(inp):
        rgb_path, output_dir = inp
        out_rgb_path = osp.join(output_dir, 'images', osp.basename(rgb_path))
        out_depth_path = osp.join(output_dir, 'depths',osp.splitext(osp.basename(rgb_path))[0] + '.png')
        out_segmenation_path = osp.join(output_dir, 'segmentations',osp.splitext(osp.basename(rgb_path))[0] + '.png')
        if osp.exists(out_rgb_path) and osp.exists(out_depth_path) and osp.exists(out_segmenation_path):
            return 
        rgb, depth, conf_img, _ = pair_rgb_depth_from_r3d(rgb_path)
        depth = (depth * 1000).astype(np.uint16)
        # valid_mask = np.logical_and(depth > 0, depth < 1000)
        # valid_mask = np.logical_and(valid_mask, conf_img == 2)
        # depth[~valid_mask] = 0

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

    path_r3ds = glob(osp.join(args.input_dir, '*.r3d'))
    for path_r3d in path_r3ds:
        try:
            fn = osp.splitext(osp.basename(path_r3d))[0]
            unzip_dir = osp.join('.cache/unzip_r3d/', fn)
            
            os.makedirs(unzip_dir, exist_ok=True)
            
            if len(glob(osp.join(unzip_dir, '*'))) == 0:
                os.system(f'unzip {path_r3d} -d {unzip_dir}')

            rgb_paths = glob(osp.join(unzip_dir, 'rgbd/*.jpg'))
            assert len(rgb_paths) > 0, f'No rgb images in {unzip_dir}'
            output_dir = osp.join(args.output_dir, fn)
            pair_inputs = [(rgb_path, output_dir) for rgb_path in rgb_paths]
            multi_thread(f, pair_inputs)
            intrinsics = pair_rgb_depth_from_r3d(rgb_paths[-1])[-1]
            mmcv.dump(intrinsics, osp.join(output_dir, 'intrinsics.json'))

            # Ask user to remove unzip_dir
            print(f'Please remove {unzip_dir} if you want to save disk space')
        except:
            print(f'Error in {path_r3d}')
            # Clean
            os.system(f'rm -rf {unzip_dir}')
            continue