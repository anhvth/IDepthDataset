import cv2
import liblzfse  # https://pypi.org/project/pyliblzfse/
import numpy as np
from avcv.all import *

rear_intrinsics = dict(
    fx=711.66571045,
    fy=711.66571045,
    cx=353.33404541,
    cy=480.73724365234375,
)
front_intrinsics = dict(
    fx=711.66571045,
    fy=711.66571045,
    cx=353.33404541,
    cy=480.73724365234375,
)


def _depth_map_to_point_cloud(depth_map, fx, fy, cx, cy):
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

    # Stack the x, y, and z arrays to create the point cloud
    point_cloud = np.stack((x, y, z), axis=-1)

    return point_cloud


def load_depth(filepath, meta):
    with open(filepath, 'rb') as depth_fh:
        raw_bytes = depth_fh.read()
        # print(raw_bytes)
        decompressed_bytes = liblzfse.decompress(raw_bytes)
        depth_img = np.frombuffer(decompressed_bytes, dtype=np.float32)

    depth_img = depth_img.reshape((meta['dh'], meta['dw']))  # For a LiDAR 3D Video

    return depth_img


def load_conf(filepath, meta):
    with open(filepath, 'rb') as depth_fh:
        raw_bytes = depth_fh.read()
        # print(raw_bytes)
        decompressed_bytes = liblzfse.decompress(raw_bytes)
        depth_img = np.frombuffer(decompressed_bytes, dtype=np.uint8)

    depth_img = depth_img.reshape((meta['dh'], meta['dw']))  # For a LiDAR 3D Video
    return depth_img


# default_meta = json.load(open('./data/r3d/metadata'))


def pcd_from_r3d(file_path, rgb_ext='jpg', depth_ext='depth', conf_ext='conf', conf_lvl=2, verbose=False):
    dir_name = os.path.dirname(file_path)
    file_name = get_name(file_path)
    rgb_path = os.path.join(dir_name, file_name + '.' + rgb_ext)
    depth_path = os.path.join(dir_name, file_name + '.' + depth_ext)
    conf_path = os.path.join(dir_name, file_name + '.' + conf_ext)

    # Get camera intrinsics
    meta_path = os.path.join(dir_name, '../metadata')
    meta = json.load(open(meta_path))
    K = np.array(meta['K']).reshape(3, 3).T
    if verbose:
        print(K)
    fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]
    # import ipdb; ipdb.set_trace()
    # Get images/point cloud
    rgb = cv2.imread(rgb_path)[..., ::-1].copy()
    
    depth_img = load_depth(depth_path, meta)
    if osp.exists(conf_path):
        conf_img = load_conf(conf_path, meta)

    depth_img = mmcv.imresize_like(depth_img, rgb, interpolation='nearest')
    
    if osp.exists(conf_path):
        conf_img = mmcv.imresize_like(conf_img, rgb, interpolation='nearest')
        valid_mask = conf_img >= conf_lvl


    xyz = _depth_map_to_point_cloud(depth_img, fx, fy, cx, cy)
    # Filter out points with low confidence
    if osp.exists(conf_path):
        valid_mask = valid_mask.reshape(-1)
        xyz = xyz.reshape(-1, 3)[valid_mask]
        rgb = rgb.reshape(-1, 3)[valid_mask]

    from idd.rgbd import pcd_from_xyz
    pcd = pcd_from_xyz(xyz, rgb)

    return pcd


# # def load_r3d(depth_filepath=None, rgb_path=None, conf_path=None, meta=default_meta, min_conf=2, reshape=True, pred_depth_map=None):
# #     """
# #     Load RGB-D data from a .depth file"""
# #     if rgb_path is None:
# #         rgb_path = depth_filepath.replace('.depth', '.jpg')
# #     if conf_path is None:
# #         conf_path = depth_filepath.replace('.depth', '.conf')

# #     if pred_depth_map is not None:
# #         depth_img_pred = pred_depth_map.copy()

# #     if depth_filepath is not None:
# #         depth_img_gt = load_depth(depth_filepath).copy()

# #     rgb = cv2.imread(rgb_path)[...,::-1].copy()
# #     depth_img_pred = mmcv.imresize_like(depth_img_pred, rgb, interpolation='nearest')
# #     depth_img_gt = mmcv.imresize_like(depth_img_gt, rgb, interpolation='nearest')
# #     conf = load_conf(conf_path)
# #     conf = mmcv.imrescale(conf, (depth_img_pred.shape[1], depth_img_pred.shape[0]), interpolation='nearest')

# #     invalid_mask = conf<min_conf
# #     depth_img_pred[invalid_mask] = 0
# #     depth_img_gt[invalid_mask] = 0

# #     # depth_img = mmcv.imrescale(depth_img, (meta['w'], meta['h']), interpolation='nearest')
# #     conf = mmcv.imrescale(conf, (meta['w'], meta['h']), interpolation='nearest')
# #     xyz_pred = _depth_map_to_point_cloud(depth_img_pred, meta['h'], meta['w'])

# #     if reshape:
# #         rgb = rgb.reshape(-1, 3)/255.
# #         xyz_pred = xyz_pred.reshape(-1, 3)
# #         xyz_gt = xyz_gt.reshape(-1, 3)

# #     return rgb, xyz_pred, invalid_mask, depth_img_pred


# def preload_pcd(path):
#     rgb, xyz, conf = load_r3d(path)
#     conf = conf.reshape(-1)
#     rgb = rgb[conf>=1]
#     xyz = xyz[conf>=1]
#     return xyz, rgb

# if __name__ == '__main__':
#     import open3d as o3d
#     from avcv.all import *
#     paths = sorted(glob('./data/r3d/rgbd/*.depth'), key=lambda path:int(get_name(path)))
#     meta = json.load(open('./data/r3d/metadata'))
#     multi_thread(preload_pcd, paths, max_workers=8)

#     vis = o3d.visualization.Visualizer()
#     vis.create_window()
#     pcd = o3d.geometry.PointCloud()
#     for i, path in enumerate(paths):
#         xyz, rgb = preload_pcd(path)
#         pcd.points = o3d.utility.Vector3dVector(xyz)
#         pcd.colors = o3d.utility.Vector3dVector(rgb)
#         if i == 0:
#             vis.add_geometry(pcd)
#         else:
#             vis.update_geometry(pcd)
#         vis.poll_events()
#         vis.update_renderer()
