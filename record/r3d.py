import numpy as np

import numpy as np
import cv2
import liblzfse  # https://pypi.org/project/pyliblzfse/
from avcv.all import *


import numpy as np
rear_intrinsics = dict(
    fx=711.66571045,
    fy=711.66571045,
    cx=353.33404541,
    cy=480.73724365234375,
)

def depth_map_to_point_cloud(depth_map, rows, cols, cam='rear'):
    if cam == 'rear':
        fx = rear_intrinsics['fx']
        fy = rear_intrinsics['fy']
        cx = rear_intrinsics['cx']
        cy = rear_intrinsics['cy']
    else:
        # Assume front camera if cam is not 'rear'
        fx = front_intrinsics['fx']
        fy = front_intrinsics['fy']
        cx = front_intrinsics['cx']
        cy = front_intrinsics['cy']

    # Create arrays to hold the x, y, and z coordinates
    x = np.empty((rows, cols), dtype=np.float32)
    y = np.empty((rows, cols), dtype=np.float32)
    z = np.empty((rows, cols), dtype=np.float32)

    # Calculate the 3D coordinates for each pixel
    x = (np.arange(cols) - cx) * depth_map / fx
    y = (np.arange(rows)[:, np.newaxis] - cy) * depth_map / fy
    z = depth_map 

    # Stack the x, y, and z arrays to create the point cloud
    point_cloud = np.stack((x, y, z), axis=-1)

    return point_cloud


def load_depth(filepath):
    with open(filepath, 'rb') as depth_fh:
        raw_bytes = depth_fh.read()
        # print(raw_bytes)
        decompressed_bytes = liblzfse.decompress(raw_bytes)
        depth_img = np.frombuffer(decompressed_bytes, dtype=np.float32)

    depth_img = depth_img.reshape((256, 192))  # For a LiDAR 3D Video

    return depth_img

def load_conf(filepath):
    with open(filepath, 'rb') as depth_fh:
        raw_bytes = depth_fh.read()
        # print(raw_bytes)
        decompressed_bytes = liblzfse.decompress(raw_bytes)
        depth_img = np.frombuffer(decompressed_bytes, dtype=np.uint8)

    depth_img = depth_img.reshape((256, 192))  # For a LiDAR 3D Video
    return depth_img

# @memoize
def load_r3d(depth_filepath, meta=json.load(open('./data/r3d/metadata')), min_conf=2, reshape=True):
    """
    Load RGB-D data from a .depth file"""
    depth_img = load_depth(depth_filepath).copy()
    rgb = cv2.imread(depth_filepath.replace('.depth', '.jpg'))[...,::-1].copy()
    conf = load_conf(depth_filepath.replace('.depth', '.conf'))
    conf_mask = conf<min_conf
    depth_img[conf_mask] = 0
    depth_img = mmcv.imrescale(depth_img, (meta['w'], meta['h']), interpolation='nearest')
    conf = mmcv.imrescale(conf, (meta['w'], meta['h']), interpolation='nearest')
    xyz = depth_map_to_point_cloud(depth_img, meta['h'], meta['w'])
    if reshape:
        rgb = rgb.reshape(-1, 3)/255.
        xyz = xyz.reshape(-1, 3)
    return rgb, xyz, conf_mask, depth_img


def preload_pcd(path):
    rgb, xyz, conf = load_r3d(path)
    conf = conf.reshape(-1)
    rgb = rgb[conf>=1]
    xyz = xyz[conf>=1]
    return xyz, rgb

if __name__ == '__main__':
    import open3d as o3d
    from avcv.all import *
    paths = sorted(glob('./data/r3d/rgbd/*.depth'), key=lambda path:int(get_name(path)))
    meta = json.load(open('./data/r3d/metadata'))
    vis = o3d.visualization.Visualizer()
    vis.create_window()
    



    multi_thread(preload_pcd, paths, max_workers=8)
    pcd = o3d.geometry.PointCloud()
    for i, path in enumerate(paths):
        xyz, rgb = preload_pcd(path)
        pcd.points = o3d.utility.Vector3dVector(xyz)
        pcd.colors = o3d.utility.Vector3dVector(rgb)
        if i == 0:
            vis.add_geometry(pcd)
        else:
            vis.update_geometry(pcd)
        vis.poll_events()
        vis.update_renderer()