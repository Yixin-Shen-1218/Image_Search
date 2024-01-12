from pathlib import Path

import cv2
import numpy
import numpy as np
from sklearn.cluster import KMeans


# SIFT cluster function, use SIFT to extract the features of each img,
# and then run KNN to get the centers as words of vocabulary
def SIFT_Cluster(number_of_center):
    SIFT = cv2.xfeatures2d_SIFT.create()
    des_list = []
    des_matrix = np.zeros((1, 128))

    for img_path in sorted(Path("static/img").glob("*.jpg")):
        print(img_path)
        img = cv2.imread(f'{img_path}')
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        kp, des = SIFT.detectAndCompute(gray, None)
        print(des.shape)
        if des is not None:
            des_matrix = np.row_stack((des_matrix, des))
        des_list.append(des)

    print('des_matrix', des_matrix)
    print(des_matrix.shape)

    des_matrix = des_matrix[1:, :]
    print(des_matrix.shape)

    print(len(des_list))

    K_Means = KMeans(n_clusters=number_of_center, random_state=14)
    K_Means.fit(des_matrix)
    centres = K_Means.cluster_centers_

    print(centres.shape)

    return centres, des_list


# transform the des into img features, the feature number of each img is the number of centres
def des2feature(des, centres_number, centres):
    img_feature_vec = np.zeros((1, centres_number), 'float32')
    for i in range(des.shape[0]):
        feature_k_rows = np.ones((centres_number, 128), 'float32')
        feature = des[i]
        feature_k_rows = feature_k_rows * feature
        feature_k_rows = np.sum((feature_k_rows - centres) ** 2, 1)
        index = np.argmax(feature_k_rows)
        img_feature_vec[0][index] += 1
    return img_feature_vec


# search the nearest images by the query img in the database
def img_Retrieve(query_path, image_features, centres):
    SIFT = cv2.xfeatures2d_SIFT.create()
    max_num = 10
    img = cv2.imread(f'{query_path}')
    img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    kp, des = SIFT.detectAndCompute(img, None)
    feature = des2feature(des, 1024, centres)
    dists, nearest_index = get_Nearest(feature, image_features, max_num)

    return dists, nearest_index


# calculate the distances between the query img and the images in the database,
# return the distances and the nearest images index in the database
def get_Nearest(feature, image_features, max_num):
    features = np.ones((image_features.shape[0], len(feature)), 'float32')
    features = features * feature
    dists = np.sum((features - image_features) ** 2, 1)
    dist_index = np.argsort(dists)[:max_num]

    return dists, dist_index


if __name__ == "__main__":
    # save the centres and des_list of the dataset
    number_of_words = 1024

    Centres, Des_List = SIFT_Cluster(number_of_words)
    np.save("static/SIFT_centres/SIFT_centres.npy", Centres)

    # save the feature vectors of the images in the database that extracted by SIFT+BOW algorithm
    i = 0
    for img_path in sorted(Path("static/img").glob("*.jpg")):
        feature_vector = des2feature(Des_List[i], number_of_words, Centres)
        print(feature_vector)
        print(feature_vector.shape)
        i = i + 1
        feature_path = Path("static/SIFT_feature") / (img_path.stem + ".npy")
        np.save(f"{feature_path}", feature_vector)
