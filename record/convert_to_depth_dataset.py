"""
    This code demonstrates how to open an r3d file and display it with open3d
"""

# install liblzfse with pip install pyliblzfse open3d
from r3d import load_r3d
# import open3d as o3d
from glob import glob
from avcv.all import *
if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('input_dir')
    parser.add_argument('output_dir')
    parser.add_argument('--conf', type=float, default=1, help='minimum confidence')
    args = parser.parse_args()

    # Clean output_dir
    if os.path.exists(args.output_dir):
        import shutil
        shutil.rmtree(args.output_dir)

    # Create a images and depth directory in side output_dir
    os.makedirs(args.output_dir + '/images', exist_ok=True)
    os.makedirs(args.output_dir + '/depths', exist_ok=True)

    # for input in tqdm(glob(args.input_dir + '/*.depth')):
    def f(input):
        rgb, xyz, conf, depth_img = load_r3d(input, min_conf=args.conf, reshape=False)
        # Clip depth in range [0, 3] 
        depth_img = np.clip(depth_img, 0, 3)
        # Normalize depth to [0, 255]
        depth_img = (depth_img / 3 * 255).astype(np.uint8)
        # Save to to depth directory
        cv2.imwrite(args.output_dir + '/depths/' + os.path.basename(input).replace('.depth', '.png'), depth_img)
        # Save rgb to images directory
        cv2.imwrite(args.output_dir + '/images/' + os.path.basename(input).replace('.depth', '.png'), rgb[...,::-1])
    multi_thread(f, glob(args.input_dir + '/*.depth'), max_workers=8)