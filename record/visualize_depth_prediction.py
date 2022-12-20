"""
        This code demonstrates how to open an r3d file and display it with open3d
"""

# install liblzfse with pip install pyliblzfse open3d
from r3d import load_r3d, rear_intrinsics, load_depth
import open3d as o3d
from avcv.all import *


import cv2
import mediapipe as mp
import numpy as np
from mediapipe_model import fmodel

mp_drawing = mp.solutions.drawing_utils
mp_selfie_segmentation = mp.solutions.selfie_segmentation


class SelfieSegmentation:
    def __init__(self, model_selection=0, bg_color=(0, 0, 0), mask_color=(255, 255, 255)):
        self.model_selection = model_selection
        self.bg_color = bg_color
        self.mask_color = mask_color

    def segment(self, image):
        with mp_selfie_segmentation.SelfieSegmentation(self.model_selection) as selfie_segmentation:
            #   for idx, file in enumerate(image_files):
            # image = cv2.imread(file)
            image_height, image_width, _ = image.shape
            results = selfie_segmentation.process(
                cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

            condition = np.stack(
                (results.segmentation_mask,) * 3, axis=-1) > 0.1
            # Generate solid color images for showing the output selfie segmentation mask.
            fg_image = np.zeros(image.shape, dtype=np.uint8)
            fg_image[:] = self.mask_color
            bg_image = np.zeros(image.shape, dtype=np.uint8)
            bg_image[:] = self.bg_color
            output_image = np.where(condition, fg_image, bg_image)
        return output_image


def draw_contour(img, points, color, thickness=1):
    # Draw contour
    points_convex = cv2.convexHull(points)
    cv2.polylines(img, [points_convex], True, color, thickness)


def get_mean_depth(depth, points):
    points_convex = cv2.convexHull(points)
    # Create a zero mask and fill the convex hull with 1
    mask = np.zeros(depth.shape, dtype=np.uint8)
    cv2.fillConvexPoly(mask, points_convex, 1)
    # Get the mean depth of the convex hull
    mean_depth = np.mean(depth[mask == 1])
    for x, y in points:
        cv2.circle(mask, (int(x), int(y)), 3, 1, -1)
    mmcv.imwrite(mask * 255, 'mask.png')
    return mean_depth


def pixel_coordinate_to_camera_coordinate(x, y, depth, intrinsic):
    """
    Convert pixel coordinate to camera coordinate.
    Args:
            x (int): x coordinate in pixel.
            y (int): y coordinate in pixel.
            depth (float): depth value.
            intrinsic (np.ndarray): camera intrinsic matrix.
    Returns:
            np.ndarray: 3D coordinate in camera coordinate.
    """
    fx, fy, cx, cy = intrinsic
    x = (x - cx) * depth / fx
    y = (y - cy) * depth / fy
    return np.array([x, y, depth])


# @memoize
def create_ply(file_name, apply_face_model=True):
    depth_path_predict = os.path.join(args.depth_dir, file_name + '.png')
    rgb_path = os.path.join(args.gt_dir, file_name + '.jpg')
    conf_path = os.path.join(args.gt_dir, file_name + '.conf')
    depth_path = os.path.join(args.gt_dir, file_name + '.depth')

    pred_depth_map = mmcv.imread(depth_path_predict, 'unchanged')
    pred_depth_map = pred_depth_map.astype(np.float32) / 255 * 1.5
    pred_depth_map = np.clip(pred_depth_map, 0, 1.5)
    bgr_img = mmcv.imread(rgb_path)
    pred_depth_map = mmcv.imresize_like(pred_depth_map, bgr_img)


    
    depth_img_gt = load_depth(depth_path)
    # import ipdb; ipdb.set_trace()
    depth_img_gt = mmcv.imresize_like(depth_img_gt, bgr_img)


    rgb, xyz, conf, _,  = load_r3d(rgb_path=rgb_path, conf_path=conf_path, min_conf=args.conf,
                             pred_depth_map=pred_depth_map, reshape=False, depth_filepath= depth_path)
    left_eye, right_eye = None, None
    if apply_face_model:
        rgb, face_landmarks_np = fmodel(bgr_img[..., ::-1].copy(), draw=False)
        def get_3d_eye(pred_depth_map):
            left_eye = face_landmarks_np[[130, 243, 23, 27], :2]
            right_eye = face_landmarks_np[[359, 463, 253, 257], :2]
            draw_contour(rgb, left_eye, (0, 0, 255), 1)
            draw_contour(rgb, right_eye, (0, 0, 255), 1)
    
            mean_depth_left_eye = get_mean_depth(pred_depth_map, left_eye)
    
            intrinsic_mat = rear_intrinsics['fx'], rear_intrinsics[
                'fy'], rear_intrinsics['cx'], rear_intrinsics['cy']
    
            left_eye_x = (left_eye[0][0] + left_eye[1][0]) / 2
            left_eye_y = (left_eye[0][1] + left_eye[1][1]) / 2
            eye_3d_left = pixel_coordinate_to_camera_coordinate(
                left_eye_x, left_eye_y, mean_depth_left_eye, intrinsic_mat)
    
            mean_depth_right_eye = get_mean_depth(pred_depth_map, right_eye)
            right_eye_x = (right_eye[0][0] + right_eye[1][0]) / 2
            right_eye_y = (right_eye[0][1] + right_eye[1][1]) / 2
            eye_3d_right = pixel_coordinate_to_camera_coordinate(
                right_eye_x, right_eye_y, mean_depth_right_eye, intrinsic_mat)
            return eye_3d_left, eye_3d_right
        
        eye_3d_left_pred, eye_3d_right_pred = get_3d_eye(pred_depth_map)
        eye_3d_left_gt, eye_3d_right_gt = get_3d_eye(depth_img_gt)


        # print('Pred Left eye 3d: {:.4f}, {:.4f}, {:.4f}'.format(
        #     eye_3d_left_pred[0], eye_3d_left_pred[1], eye_3d_left_pred[2]))
        # print('GT Left eye 3d: {:.4f}, {:.4f}, {:.4f}'.format(
        #     eye_3d_left_gt[0], eye_3d_left_gt[1], eye_3d_left_gt[2]))


        # print('Pred Right eye 3d: {:.4f}, {:.4f}, {:.4f}'.format(
        #     eye_3d_right_pred[0], eye_3d_right_pred[1], eye_3d_right_pred[2]))
        # print('GT Right eye 3d: {:.4f}, {:.4f}, {:.4f}'.format(
        #     eye_3d_right_gt[0], eye_3d_right_gt[1], eye_3d_right_gt[2]))


        # Print abs diff in cm
        print('Left eye 3d abs diff: {:.4f}, {:.4f}, {:.4f}'.format(
            np.abs(eye_3d_left_pred[0] - eye_3d_left_gt[0]) * 100,
            np.abs(eye_3d_left_pred[1] - eye_3d_left_gt[1]) * 100,
            np.abs(eye_3d_left_pred[2] - eye_3d_left_gt[2]) * 100))
        print('Right eye 3d abs diff: {:.4f}, {:.4f}, {:.4f}'.format(
            np.abs(eye_3d_right_pred[0] - eye_3d_right_gt[0]) * 100,
            np.abs(eye_3d_right_pred[1] - eye_3d_right_gt[1]) * 100,
            np.abs(eye_3d_right_pred[2] - eye_3d_right_gt[2]) * 100))





    segmenter = SelfieSegmentation()
    out = segmenter.segment(rgb)

    # import ipdb; ipdb.set_trace()

    mask = out[:, :, 0] > 0
    # import ipdb; ipdb.set_trace()
    # mask = cv2.morphologyEx(mask.astype(np.uint8), cv2.MORPH_ERODE, np.ones((50, 50)))
    # mask = mask.astype('bool')
    # import ipdb; ipdb.set_trace()

    xyz = xyz[mask]
    rgb = rgb[mask]

    # Flatten xyz and rgb
    xyz = xyz.reshape(-1, 3)
    rgb = rgb.reshape(-1, 3)
    rgb = rgb.astype(np.float32) / 255

    return rgb, xyz, eye_3d_left_pred, eye_3d_right_pred


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--file_name', '-f', default=0,
                        type=int, help='input file name (without extension')
    parser.add_argument('--depth_dir', default="output/depth",
                        help='directory of depth predictions')
    parser.add_argument('--conf', type=float, default=0.5,
                        help='minimum confidence')
    parser.add_argument('--gt_dir', default="data/r3d/rgbd",
                        help='directory of ground truth')
    args = parser.parse_args()

    view_control = o3d.visualization.ViewControl()
    view_control.set_lookat([0, 0, 0])

    if args.file_name >= 0:
        for i in range(args.file_name, args.file_name+1000, 30):
            try:
                rgb, xyz, left_eye, right_eye = create_ply(str(i))
                pcd = o3d.geometry.PointCloud()
                pcd.points = o3d.utility.Vector3dVector(xyz)
                pcd.colors = o3d.utility.Vector3dVector(rgb)

                line_set = o3d.geometry.LineSet()
                # print("left_eye: ", left_eye)
                # print("right_eye: ", right_eye)
                line_set.points = o3d.utility.Vector3dVector(
                    [[0, 0, 0], left_eye.tolist(), right_eye.tolist()])
                line_set.colors = o3d.utility.Vector3dVector(
                    [[1, 0, 0] for _ in range(2)])

                lines = o3d.utility.Vector2iVector([[0, 1], [0, 2]])
                line_set.lines = lines
                # Draw the point cloud and visualize with eye at origin
                o3d.visualization.draw_geometries([pcd, line_set])
            except:
                pass

    else:
        inputs = os.listdir(args.depth_dir)
        inputs = [int(get_name(input)) for input in inputs]
        inputs.sort()
        inputs = [str(input) for input in inputs]

        vis = o3d.visualization.Visualizer()
        #

        vis.create_window(view_control=view_control)
        pcd = o3d.geometry.PointCloud()

        multi_thread(create_ply, inputs, max_workers=7,
                     desc="Creating point clouds")
        for i, input in tqdm(enumerate(inputs)):
            rgb, xyz = create_ply(input)

            pcd.points = o3d.utility.Vector3dVector(xyz)
            pcd.colors = o3d.utility.Vector3dVector(rgb)
            if i == 0:
                vis.add_geometry(pcd)
            else:
                vis.update_geometry(pcd)
            vis.poll_events()
            vis.update_renderer()
