import cv2
from avcv.all import *
# Set up the object points in the 3D world and the corresponding points in the images
object_points = []
image_points = []

# Set the number of inner corners per row and column in the checkerboard
pattern_size = (9, 6)

# Load the checkerboard images
# images = [cv2.imread('image1.jpg'), cv2.imread('image2.jpg'), cv2.imread('image3.jpg')]
def read_heic(f):
    from PIL import Image
    import pillow_heif

    heif_file = pillow_heif.read_heif(f)
    image = Image.frombytes(
        heif_file.mode,
        heif_file.size,
        heif_file.data,
        "raw",
    )
    img = np.array(image)
    img = cv2.resize(img, (640, 480))
    return img

import argparse
parser = argparse.ArgumentParser()
parser.add_argument('input_dir')
args = parser.parse_args()
images = glob(args.input_dir+'/*.HEIC')
images = [read_heic(i) for i in images]

# Loop over the images
objp = np.zeros((pattern_size[0]*pattern_size[1], 3), np.float32)
objp[:,:2] = np.mgrid[0:pattern_size[0], 0:pattern_size[1]].T.reshape(-1, 2)
for img in images:
    # Convert the image to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Find the corners of the checkerboard in the image
    ret, corners = cv2.findChessboardCorners(gray, pattern_size, None)
    # If the corners are found, add the object points and image points for the image
    if ret:
        object_points.append(objp)
        image_points.append(corners)

# Obtain the camera matrix and distortion coefficients
# import ipdb; ipdb.set_trace()
# Calib using opencv
ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(object_points, image_points, (640, 480), None , None)
print('ret', ret)
print('mtx', mtx)
print('dist', dist)
print('rvecs', rvecs)
print('tvecs', tvecs)

