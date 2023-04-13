import os
import numpy as np
import matplotlib.pyplot as plt
from random import choice
from scipy.optimize import linprog


def get_sparse_representation(A_con, c, bounds, test_sample, eps, iteration):
    y_con = np.concatenate((test_sample+eps, -test_sample+eps), axis=0)  # set constrain, max noise term energy = eps
    res = linprog(c=c, A_ub=A_con, b_ub=y_con, bounds=bounds, method='highs')
    sparse_x = res.x
    #  print(res.status)
    # if res.status == 0 and iteration == 0:
    #     plt.plot(sparse_x)
    #     plt.show()
    #     plt.title("example of sparse representation")
    return sparse_x


def src_decision(A, sample_per_class, sparse_x, class_num, test_sample):
    pred_error = []
    for c in range(class_num):
        coeff = sparse_x[c*sample_per_class:sample_per_class*(c+1)]
        pred_y = np.dot(A[:, c*sample_per_class:sample_per_class*(c+1)], coeff)
        pred_error.append(np.linalg.norm(pred_y-test_sample, 2))
    decision = pred_error.index(min(pred_error))
    return decision


def vote(decision_list):
    freq = {}
    for d in decision_list:
        if d in freq:
            freq[d] += 1
        else:
            freq[d] = 1
    max_freq = max(freq.values())
    return [key for key, value in freq.items() if value == max_freq]


def classification(train_set, test_set, eps):
    class_num, train_sample_per_class = train_set.shape[0], train_set.shape[1]
    p, count = 0, 0

    A = [train_set[c, ] for c in range(class_num)]
    A = np.concatenate(A, axis=0).transpose((1, 0))

    # Define the objective function and constraints for the linear program: y = Ax + z, ||z||_2 < eps
    A_con = np.concatenate((A, -A), axis=0)
    c = np.ones(A.shape[1])
    bounds = [(0, None) for i in range(A.shape[1])]

    print("start classification")
    for role in range(len(test_set)):
        print("testing class "+str(role))
        for i in range(test_set[role].shape[0]):
            sparse_x = get_sparse_representation(A_con, c, bounds, test_set[role][i, :], eps, i)
            if sparse_x is not None:
                decision = src_decision(A, train_sample_per_class, list(sparse_x), class_num, test_set[role][i, :])
            else:
                decision = 0
            p += 1 if decision == role else 0
            count += 1
    acc = p / count
    print("accuracy = "+str(acc*100)+"%")


def block_wise_classification(train_set, test_set, eps):
    train_blocks = [train_set[:, :, i, :] for i in range(train_set.shape[2])]
    train_set = np.concatenate(train_blocks, axis=1)
    class_num, train_sample_per_class = train_set.shape[0], train_set.shape[1]
    p, count = 0, 0
    #
    A = [train_set[c,] for c in range(class_num)]
    # A = [m/np.sqrt(np.sum(m**2)) for m in A]
    A = np.concatenate(A, axis=0).transpose((1, 0))
    #
    # # Define the objective function and constraints for the linear program: y = Ax + z, ||z||_2 < eps
    A_con = np.concatenate((A, -A), axis=0)
    c = np.ones(A.shape[1])
    bounds = [(0, None) for i in range(A.shape[1])]
    #
    # print("start classification")
    for role in range(len(test_set)):
        print("testing class "+str(role))
        for i in range(test_set[role].shape[0]):
            decisions = []
            for block in range(test_set[role].shape[1]):
                sparse_x = get_sparse_representation(A_con, c, bounds, test_set[role][i, block, :], eps, i)
                if sparse_x is not None:
                    decisions.append(src_decision(A, train_sample_per_class, list(sparse_x), class_num, test_set[role][i, block, :]))
                else:
                    decisions.append(0)
            # print(decisions, vote(decisions))
            decision = choice(vote(decisions))
            print("True class: "+str(role), "Predicted class: "+str(decision))
            p += 1 if decision == role else 0
            count += 1
    acc = p / count
    print("accuracy = "+str(acc*100)+"%")


def visualize_wine_feature(data_with_classes, axis1, axis2):
    colors = ['g', 'b', 'r']
    for ls in range(len(data_with_classes)):  # check number of different class: 59, 71, 48 respectively
        reduced_feature = [data_with_classes[ls][:, axis1], data_with_classes[ls][:, axis2]]  # dimension to be visualized
        for p in range(reduced_feature[0].shape[0]):
            plt.scatter(reduced_feature[0][p], reduced_feature[1][p], c=colors[ls])
    plt.show()


if __name__ == "__main__":
    data_array = np.load('../matrix_data/data_array.npy') * 100
    block_array = np.load('../matrix_data/block_array.npy') * 100
    #  reduce to 3 classes: Darth Vader, Green Goblin, Thanos
    data_array = np.concatenate([data_array[:2, ], np.expand_dims(data_array[3, :, :], axis=0)], axis=0)
    block_array = np.concatenate([block_array[:2, ], np.expand_dims(block_array[3, :, :, :], axis=0)], axis=0)
    print(data_array.shape, block_array.shape)
    train_set = data_array[:, :15, :]
    test_set = data_array[:, 15:, :]
    train_block_set = block_array[:, :15, :, :]
    test_block_set = block_array[:, 15:, :, :]
    # classification(train_set, test_set)
    # block_wise_classification(train_block_set, test_block_set)
