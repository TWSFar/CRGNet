python object-count/test_dota.py --checkpoint /home/twsf/work/CRGNet/object-count/run/DOTA/64_1_1/last.pth.tar
python density_tools/generate_test_chips.py
python mmdetection/tools_dota/cocotest.py
python density_tools/metric.py

python object-count/test_dota.py --checkpoint /home/twsf/work/CRGNet/object-count/run/DOTA/64_1_7/last.pth.tar
python density_tools/generate_test_chips.py
python mmdetection/tools_dota/cocotest.py
python density_tools/metric.py

python object-count/test_dota.py --checkpoint /home/twsf/work/CRGNet/object-count/run/DOTA/64_1_20/last.pth.tar
python density_tools/generate_test_chips.py
python mmdetection/tools_dota/cocotest.py
python density_tools/metric.py

python object-count/test_dota.py --checkpoint /home/twsf/work/CRGNet/object-count/run/DOTA/64_1_100/last.pth.tar
python density_tools/generate_test_chips.py
python mmdetection/tools_dota/cocotest.py
python density_tools/metric.py