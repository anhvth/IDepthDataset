# Visualize sequence of ply files using open3d
# Path: tools/vis_seq_ply.py

import open3d as o3d
import argparse
import os
from glob import glob
from idd.rgbd import visualize_rgbd
import time
import numpy as np
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--ply_dir', type=str, default='data/rgbd/ply', help='ply dir')
    args = parser.parse_args()
    path_ply = glob(f'{args.ply_dir}/*.ply')
    path_ply.sort(key=lambda x: int(os.path.basename(x).split('.')[0]))

    vis = o3d.visualization.Visualizer()
    vis.create_window()
    # Position the camera so that the point cloud is in the center of the view
    def load_pcd(path):
        pcd = o3d.io.read_point_cloud(path)
        pcd = pcd.select_by_index(np.where(np.asarray(pcd.points)[:, 2] < 800)[0])
        return pcd
    ctr = vis.get_view_control()
    ctr.set_front([0, 0, -1])
    ctr.set_lookat([0, 0, 0])
    ctr.set_up([0, -1, 0])
    ctr.set_zoom(0.5)

    vis.get_render_option().point_size = 1

    _pcd = load_pcd(path_ply[0])
    vis.add_geometry(_pcd)
    for i, ply_path in enumerate(path_ply):
        # pcd = o3d.io.read_point_cloud(ply_path)
        pcd = load_pcd(ply_path)
        _pcd.points = pcd.points
        _pcd.colors = pcd.colors
        vis.update_geometry(_pcd)
        vis.poll_events()
        vis.update_renderer()
        print('Processing {}/{}'.format(i, len(path_ply)), '| ', ply_path)
        vis.run()
