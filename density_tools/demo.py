import os


train_list = []
train_file = "/home/twsf/data/Visdrone/density_chip/ImageSets/Main/train.txt"
with open(train_file, 'r') as f:
    for line in f.readlines():
        train_list.append(line.strip())


test_list = []
test_file = "/home/twsf/data/Visdrone/density_chip/ImageSets/Main/test.txt"
with open(test_file, 'r') as f:
    for line in f.readlines():
        test_list.append(line.strip())

train_test_list = train_list + test_list
new_file = "/home/twsf/data/Visdrone/density_chip/ImageSets/Main/traintest.txt"
with open(new_file, 'w') as f:
    for line in train_test_list:
        f.writelines(line+'\n')
