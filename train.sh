# python density_tools/generate_masks.py
# python density_tools/regress/statistics.py
# python density_tools/regress/statistics2.py
# python density_tools/regress/gbm.py
python density_tools/generate_chips.py
# python density_tools/tools/xml2json_chip.py
cd mmdetection
./mmdetection/tools_visdrone/dist_train.sh ./tools_visdrone/configs/density/ATSS_r101.py 2 --validate
