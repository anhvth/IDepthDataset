"""
    This code demonstrates how to open an r3d file and display it with open3d
"""

# install liblzfse with pip install pyliblzfse open3d
from r3d import load_r3d
import open3d as o3d
if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('input')
    parser.add_argument('--conf', type=float, default=0.5, help='minimum confidence')
    args = parser.parse_args()


    rgb, xyz, conf = load_r3d(args.input, min_conf=args.conf)
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(xyz)
    pcd.colors = o3d.utility.Vector3dVector(rgb)
    o3d.visualization.draw_geometries([pcd])