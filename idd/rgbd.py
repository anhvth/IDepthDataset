import numpy as np
import open3d as o3d
import mmcv

def pcd_from_xyz(xyz, rgb=None):
    """
    Create open3d point cloud from xyz and rgb."""
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(xyz.reshape(-1, 3))
    if rgb is not None:
        rgb = rgb.reshape(-1, 3)
        if rgb.max() > 1:
            rgb = rgb / 255
        pcd.colors = o3d.utility.Vector3dVector(rgb)
    return pcd

def depth_map_to_pcd(depth_map, fx, fy, cx, cy, rgb=None):
    """
    Convert depth map to point cloud.
    """
    rows, cols = depth_map.shape[:2]
    # Create arrays to hold the x, y, and z coordinates
    x = np.empty((rows, cols), dtype=np.float32)
    y = np.empty((rows, cols), dtype=np.float32)
    z = np.empty((rows, cols), dtype=np.float32)
    # Calculate the 3D coordinates for each pixel
    x = (np.arange(cols) - cx) * depth_map / fx
    y = (np.arange(rows)[:, np.newaxis] - cy) * depth_map / fy
    z = depth_map
    xyz = np.stack((x, y, z), axis=-1)
    pcd = pcd_from_xyz(xyz, rgb)
    return pcd

def mask_pcd(pcd, max_depth=None):
    """
    Mask point cloud.
    """
    xyz = np.asarray(pcd.points)
    rgb = np.asarray(pcd.colors)
    if max_depth is not None:
        mask = xyz[:, 2] < max_depth
        xyz = xyz[mask]
        rgb = rgb[mask]
    pcd = pcd_from_xyz(xyz, rgb)
    return pcd

def visualize_rgbd(depth, fx, fy, cx, cy, rgb=None, max_depth=None):
    """
    Visualize rgbd images and depth images.
    """
    if rgb is not None:
        rgb = mmcv.imread(rgb, channel_order='rgb')

    pcd = depth_map_to_pcd(depth, fx, fy, cx, cy, rgb)
    pcd = mask_pcd(pcd, max_depth)
    o3d.visualization.draw_geometries([pcd])
