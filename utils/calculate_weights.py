import os
from tqdm import tqdm
import numpy as np
import torch


def calculate_weights_labels(dataloader, num_classes, root_dir):
    classes_weights_file = os.path.join(root_dir + '_classes_weights.npy')

    if os.path.isfile(classes_weights_file):
        weight = np.load(classes_weights_file)
    else:
        # Create an instance from the data loader
        z = np.zeros((num_classes,))
        # Initialize tqdm
        tqdm_batch = tqdm(dataloader)
        print('Calculating classes weights')
        for sample in tqdm_batch:
            y = sample['label']
            y = y.detach().cpu().numpy()
            mask = (y >= 0) & (y < num_classes)
            labels = y[mask].astype(np.uint8)
            count_l = np.bincount(labels, minlength=num_classes)
            z += count_l
        tqdm_batch.close()
        total_frequency = np.sum(z)
        class_weights = []
        for frequency in z:
            class_weight = 1 / (np.log(1.02 + (frequency / total_frequency)))
            class_weights.append(class_weight)
        weight = np.array(class_weights)
        np.save(classes_weights_file, weight)

    return weight.astype(np.float32)
