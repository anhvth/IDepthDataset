RGB_DIR=./data/211222-2/rgbd
CONF_DIR=./training/output/segmentation
DEP_DIR=./training/output/depth
INT=training/datasets/161222/intrinsics.json
FILENAME=123
python tools/vis_rgbd.py --rgb $RGB_DIR/$FILENAME.jpg --intrinsic $INT --depth $DEP_DIR/$FILENAME.png  --conf $CONF_DIR/$FILENAME.png