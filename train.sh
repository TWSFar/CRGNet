# python density_tools/generate_masks.py
# cp /home/twsf/data/Visdrone/ships/Annotations/* /home/twsf/data/Visdrone/density_chip/Annotations
# cp /home/twsf/data/Visdrone/ships/JPEGImages/* /home/twsf/data/Visdrone/density_chip/JPEGImages
# cp /home/twsf/data/Visdrone/ships/ImageSets/Main/ship.txt /home/twsf/data/Visdrone/density_chip/ImageSets/Main/


python density_tools/generate_chips.py
python density_tools/tools/xml2json_chip.py
# python density_tools/debug.py
cd mmdetection
./tools_visdrone/dist_train.sh tools_visdrone/configs/density/cascade_hrnet.py 2 --validate