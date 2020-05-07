import json

with open("/home/twsf/data/Visdrone/density_chip/Annotations_json/instances_val.json", 'r') as f:
    chip_loc = json.load(f)
# json.loads()