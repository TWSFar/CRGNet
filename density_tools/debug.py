masaic = []
with open("/home/twsf/data/UAVDT/density_chip/ImageSets/Main/mosaic.txt", 'r') as f:
    for line in f.readlines():
        masaic.append(line.strip())

with open("/home/twsf/data/UAVDT/density_chip/ImageSets/Main/train.txt", 'a') as f:
    for line in masaic:
        f.writelines(line + '\n')