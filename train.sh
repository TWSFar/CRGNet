python density_tools/generate_chips.py
python density_tools/tools/xml2json_chip.py
cd mmdetection
./tools_dota/dist_train.sh tools_dota/configs/density/ATSS_r50_fpn_giou.py 2 --validate

