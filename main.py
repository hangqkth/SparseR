import numpy as np
from utils.functionality import classification, block_wise_classification, visualize_wine_feature
import csv
import random
import matplotlib.pyplot as plt


def main(dataset):
    """main classification function"""
    if dataset == "image":
        data_array = np.load('matrix_data/data_array.npy') * 100
        block_array = np.load('matrix_data/block_array.npy') * 100
        #  reduce to 3 classes: Darth Vader, Green Goblin, Thanos
        data_array = np.concatenate([data_array[:2, ], np.expand_dims(data_array[3, :, :], axis=0)], axis=0)
        block_array = np.concatenate([block_array[:2, ], np.expand_dims(block_array[3, :, :, :], axis=0)], axis=0)
        print(data_array.shape, block_array.shape)
        train_set = data_array[:, :15, :]
        test_set = data_array[:, 15:, :]
        train_block_set = block_array[:, :15, :, :]
        test_block_set = [block_array[i, 15:, :, :] for i in range(block_array.shape[0])]
        # classification(train_set, test_set, eps=4.8)
        block_wise_classification(train_block_set, test_block_set, eps=4.8)

    elif dataset == "wine":
        csv_reader = csv.reader(open("./wine.csv"))
        data_list = [line for line in csv_reader]
        attr_list, data_list = data_list[0], data_list[1:]
        data_with_classes = [[], [], []]
        for i in range(len(data_list)):
            data_list[i] = [float(element) for element in data_list[i]]
            data_with_classes[int(data_list[i][0])-1].append(data_list[i][1:])
        data_with_classes = [np.array(lst) for lst in data_with_classes]
        visualize_wine_feature(data_with_classes, 0, 11)
        train_sample_per_class = 35  # total number per class: 59, 71, 48
        train_set = np.stack([matrix[:train_sample_per_class] for matrix in data_with_classes], axis=0)
        test_set = [matrix[train_sample_per_class:] for matrix in data_with_classes]
        classification(train_set, test_set, eps=1.3)

    else:
        print("Invalid dataset, input 'image' or 'wine'.")

print("hello")
main("image")  # input "image" for "wine"
