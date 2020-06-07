cd density_tools
python generate_chips.py
cd ../mmdetection
./tools_visdrone/dist_train.sh ./tools_visdrone/configs/density/faster_rcnn_x101.py 2 --validate