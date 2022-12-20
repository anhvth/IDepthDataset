import os.path as osp, mmcv, open3d as o3d, numpy as np, cv2, liblzfse, os, json

def get_name(filepath):
    """
    Get the name of the file.
    """
    return osp.splitext(osp.basename(filepath))[0]


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
    
    depth_img = load_depth(depth_path, meta).copy()
    if osp.exists(conf_path):
        conf_img = load_conf(conf_path, meta)
    if rgb.shape[:2] != depth_img.shape[:2]:
        if verbose:
            print('Resizing depth image to match rgb image')
        depth_img = mmcv.imresize_like(depth_img, rgb, interpolation='nearest')
    
    if osp.exists(conf_path):
        conf_img = mmcv.imresize_like(conf_img, rgb, interpolation='nearest')
        valid_mask = conf_img >= conf_lvl

    # import ipdb; ipdb.set_trace()
    depth_img[np.isnan(depth_img)] = 0

    xyz = _depth_map_to_point_cloud(depth_img, fx, fy, cx, cy)

    if osp.exists(conf_path):
        valid_mask = valid_mask.reshape(-1)
        xyz = xyz.reshape(-1, 3)[valid_mask]
        rgb = rgb.reshape(-1, 3)[valid_mask]

    from idd.rgbd import pcd_from_xyz

    pcd = pcd_from_xyz(xyz, rgb)

    return pcd


