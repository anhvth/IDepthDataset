RGB_DIR=./data/rgbd/2022-12-16--15-54-58/images
CONF_DIR=output/segmentation
DEP_DIR=output/depth
INT=$RGB_DIR/../intrinsics.json
FILENAME=999
python tools/vis_rgbd.py --rgb $RGB_DIR/$FILENAME.jpg --intrinsic $INT --depth $DEP_DIR/$FILENAME.png  --conf $CONF_DIR/$FILENAME.png