"""
    This code demonstrates how to open an r3d file and display it with open3d
"""

# install liblzfse with pip install pyliblzfse open3d
from r3d import load_r3d
# import open3d as o3d
from glob import glob
from avcv.all import *
from visualize_depth_prediction import SelfieSegmentation
if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('input_dir')
    parser.add_argument('output_dir')
    parser.add_argument('--conf', type=float, default=1, help='minimum confidence')
    args = parser.parse_args()

    # Clean output_dir
    # if os.path.exists(args.output_dir):
    #     import shutil
    #     shutil.rmtree(args.output_dir)

    # Create a images and depth directory in side output_dir
    os.makedirs(args.output_dir + '/images', exist_ok=True)
    os.makedirs(args.output_dir + '/depths', exist_ok=True)
    os.makedirs(args.output_dir + '/segmentation', exist_ok=True)

    # for input in tqdm(glob(args.input_dir + '/*.depth')):
    segmenter = SelfieSegmentation()
    def f(input):
        rgb, xyz, conf, depth_img = load_r3d(input, min_conf=args.conf, reshape=False)
        out = segmenter.segment(rgb)
        # depth >1 is invalid
        depth_img[depth_img > 1.5] = 0
        # depth_img = np.clip(depth_img, 0, 3)
        depth_img = ((depth_img / 1.5) * 255).astype(np.uint8)
        # Save to to depth directory
        cv2.imwrite(args.output_dir + '/depths/' + os.path.basename(input).replace('.depth', '.png'), depth_img)
        # Save rgb to images directory
        cv2.imwrite(args.output_dir + '/images/' + os.path.basename(input).replace('.depth', '.png'), rgb[...,::-1])
        # save out to segmentation directory
        cv2.imwrite(args.output_dir + '/segmentation/' + os.path.basename(input).replace('.depth', '.png'), out[...,::-1])
        
        
    # multi_thread(f, glob(args.input_dir + '/*.depth'), max_workers=8)

    # Combine depth and segmentation
    for path_depth in tqdm(glob(args.output_dir + '/depths/*.png')):
        depth = cv2.imread(path_depth, cv2.IMREAD_GRAYSCALE)
        seg = cv2.imread(path_depth.replace('depths', 'segmentation'), cv2.IMREAD_GRAYSCALE)
        depth[seg == 0] = 0
        out_path = path_depth.replace('depths', 'depth_segmentation')
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        cv2.imwrite(out_path, depth)
