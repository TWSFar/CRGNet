
def generate_crop_region(regions, mask, mask_shape, img_shape, gbm):
    """
    generate final regions
    enlarge regions < 300
    """
    width, height = mask_shape
    final_regions = []
    for box in regions:
        # show_image(mask, np.array(box)[None])
        mask_chip = mask[box[1]:box[3], box[0]:box[2]]
        box_w, box_h = box[2] - box[0], box[3] - box[1]
        obj_area = max(np.where(mask_chip > 0, 1, 0).sum(), 1)
        obj_num = max(mask_chip.sum(), 1.0)
        chip_area = box_w * box_h
        minweight = 351 / chip_area  # minsize is 300
        weight = gbm.predict([[obj_num, obj_area, chip_area, img_shape[0]*img_shape[1]]])[0]
        if weight <= 0.6 and (box_w > width * 0.3 or box_h > height * 0.4):
        # if weight <= 0.6 and (box_w > width * 0.3 or box_h > height * 0.3):
            # show_image(mask, np.array(box)[None])
            final_regions.extend(region_split(box, mask_shape, weight))
        elif weight >= 1 and box_w < width * 0.5 and box_h < height * 0.6:
        # elif weight >= 1 and box_w < width * 0.5 and box_h < height * 0.5:
            weight = np.clip(weight, 1, 4)
            weight = max(weight, minweight)
            final_regions.append(region_enlarge(box, mask_shape, weight))
        else:
            weight = max(weight, minweight)
            final_regions.append(region_enlarge(box, mask_shape, weight))

    final_regions = np.array(final_regions)
    # show_image(mask, final_regions)

    while(0):
        idx = np.zeros((len(final_regions)))
        for i in range(len(final_regions)):
            for j in range(len(final_regions)):
                if i == j or idx[i] == 1 or idx[j] == 1:
                    continue
                if overlap(final_regions[i], final_regions[j], thresh=0.8):
                    final_regions[i] = bbox_merge(final_regions[i], final_regions[j])
                    idx[j] = 1
        if sum(idx) == 0:
            break
        final_regions = final_regions[idx == 0]
    if len(final_regions):
        final_regions = delete_inner_region(final_regions, mask_shape)
    # show_image(mask, final_regions)
    return final_regions