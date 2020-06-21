import numpy as np
log_file = "/home/twsf/work/CRGNet/mmdetection/tools_uavdt/work_dirs/ATSS_x101_fpn_giou/20200617_194852.log"


def show(x, y, i):
    from matplotlib import pyplot as plt
    plt.plot(x, y)
    plt.savefig(f"result_analys_{i}.jpg")
    plt.show()
    plt.close()


losses = []
maps = []
maps50 = []
temp = []
flag = False
with open(log_file) as f:
    for line in f.readlines():
        if "INFO" in line and "time" in line:
            loss = float(line.split()[-1].strip())
            temp.append(loss)
            flag = True
        else:
            if len(temp) and flag:
                flag = False
                losses.append(np.mean(temp))
        if "bbox_mAP" in line:
            idx = line.split().index("bbox_mAP:")
            maps.append(float(line.split()[idx+1].strip()[:-1]))
        if "bbox_mAP_50" in line:
            idx = line.split().index("bbox_mAP_50:")
            maps50.append(float(line.split()[idx+1].strip()[:-1]))


show(np.arange(len(losses)), losses, 1)
show(np.arange(len(maps)), maps, 2)
show(np.arange(len(maps50)), maps50, 3)
