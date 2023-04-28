import os
import numpy as np
import matplotlib.image as mpimg
import cv2


def load_data_into_npy(data_source):
    """Preprocess: load image, padding, divided into blocks, down sampling, normalizing, save as .npy file"""
    class_file_list = [os.path.join(data_source, path) for path in os.listdir(data_source)]
    max_high, max_length = 276, 345  # test all images to find out, 276, 345
    data_array, block_array = [], []
    for c in range(len(class_file_list)):
        img_file_list = [os.path.join(class_file_list[c], path) for path in os.listdir(class_file_list[c])]
        # dealing with file paths
        one_class_img, one_class_block = [], []
        for i in range(len(img_file_list)):
            img = mpimg.imread(img_file_list[i])
            high, length = img.shape[0], img.shape[1]
            # max_high = high if high > max_high else max_high
            # max_length = length if length > max_length else max_length
            img = cv2.copyMakeBorder(img, (max_high-high)//2, (max_high-high)//2, (max_length-length)//2, (max_length-length)//2, cv2.BORDER_REFLECT)  # padding
            img = img[(max_high-200)//2:-(max_high-200)//2, (max_length-200)//2:-(max_length-200)//2, :]
            img = cv2.copyMakeBorder(img, 200-img.shape[0], 0, 200-img.shape[1], 0, cv2.BORDER_REFLECT)  # fixed size 200
            # img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
            blocks = [img[:100, :100, :], img[100:, 100:, :], img[:100, 100:, :], img[100:, :100, :]]
            # blocks = [img[:50, :50, :], img[:50, 50:100, :], img[:50, 100:150, :], img[:50, 150:, :],
            #           img[50:100, :50, :], img[50:100, 50:100, :], img[50:100, 100:150, :], img[50:100, 150:, :],
            #           img[100:150, :50, :], img[100:150, 50:100, :], img[100:150, 100:150, :], img[100:150, 150:, :],
            #           img[150:, :50, :], img[150:, 50:100, :], img[150:, 100:150, :], img[150:, 150:, :]]
            blocks = [cv2.pyrDown(cv2.pyrDown(cv2.pyrDown(block))) for block in blocks]
            blocks = [block.ravel()/np.linalg.norm(block.ravel(), 2) for block in blocks]
            # blocks = [block.ravel() for block in blocks]
            blocks = np.array(blocks).astype(np.float64)
            one_class_block.append(blocks)
            img = cv2.pyrDown(cv2.pyrDown(cv2.pyrDown(cv2.pyrDown(cv2.pyrDown(img)))))  # down sampling to proper size (35, 44, 3)
            img = img.astype(np.float64)
            one_class_img.append(img.ravel()/np.linalg.norm(img.ravel(), 2))  # flatten into vector and L2 normalization
            # one_class_img.append(img.ravel())
        data_array.append(one_class_img)
        block_array.append(one_class_block)
    data_array, block_array = np.array(data_array), np.array(block_array)
    print(data_array.shape)  # (5, 20, 507), 5 classes, 20 images per subject
    print(block_array.shape)  # (5, 20, 16, 507), 5 classes, 20 images divided to 16 blocks per subject
    return data_array, block_array


def show_img(img_array):
    cv2.imshow('0', img_array)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    data_source = '../image_data/'
    data_array, block_array = load_data_into_npy(data_source)
    np.save('../matrix_data/data_array.npy', data_array)
    np.save('../matrix_data/block_array.npy', block_array)
