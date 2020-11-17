import numpy as np
log_file = "/home/twsf/work/CRGNet/mmdetection/tools_uavdt/work_dirs/ATSS_r50_fpn_giou_mosaic /20201113_022632.log"
x_interval = 1
y_interval = 0.02


def show(x, y, title):
    from matplotlib import pyplot as plt
    from matplotlib.pyplot import MultipleLocator
    x_major_locator = MultipleLocator(x_interval)
    y_major_locator = MultipleLocator(y_interval)
    ax = plt.gca()
    ax.xaxis.set_major_locator(x_major_locator)
    ax.yaxis.set_major_locator(y_major_locator)
    plt.title(title)
    plt.grid()
    plt.plot(x, y)
    plt.savefig(f"analys_{title}.jpg")
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
                temp = []

        if "bbox_mAP" in line:
            idx = line.split().index("bbox_mAP:")
            maps.append(float(line.split()[idx+1].strip()[:-1]))
        if "bbox_mAP_50" in line:
            idx = line.split().index("bbox_mAP_50:")
            maps50.append(float(line.split()[idx+1].strip()[:-1]))

    if len(temp):
        losses.append(np.mean(temp))


show(np.arange(1, len(losses)+1), losses, "loss")
show(np.arange(1, len(maps)+1), maps, "map")
show(np.arange(1, len(maps50)+1), maps50, "map_50")
for v in maps:
    print(v*100 - 3.9)