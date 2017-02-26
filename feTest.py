# Test code for featureExtractor.py
# Just compile and run

import numpy as np
import cv2
import pickle
import featureExtractor as fe

kmeans_filename = '.\\Model\\kmeans'
vector_filename = '.\\Model\\vector'
doc_filename = '.\\Model\\doc_occ'
stoplist_filename = '.\\Model\\stoplist'

img_path = '.\\Groundhog day\\I_02376.jpg'
cluster_num = 200

# Load all the images and extract SIFT descriptors
# path_pool, gray, kp, des = fe.loadData('.\\Groundhog day\\', start = 60, stop = 3000, step = 30, flag = 'so')
# # Cluster the descriptors with kmeans algorithm
# kmeans = fe.clusterSKLearn(des, cluster_num, eps = 1e-2)

with open(kmeans_filename, 'rb') as kmeans_in:
    kmeans = pickle.load(kmeans_in)
best_label = kmeans.labels_
center = kmeans.cluster_centers_

# Generate stop list that contains 5% of total number of words (5% * cluster_num)
# list = fe.stopList(vector, k = 0.05)
with open(stoplist_filename, 'rb') as stoplist_in:
    stoplist = pickle.load(stoplist_in)
print 'The indexes in the stop list are: \n %s. \n' % list

with open(vector_filename, 'rb') as vector_in:
    vector = pickle.load(vector_in)
print 'There are %d     words in all the documents.' % np.sum(vector)
print 'Frequency of occurrence of each word: \n %s. \n' % np.sum(vector, axis = 0)


img = cv2.imread(img_path)
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
cv2.namedWindow('image')
cv2.imshow('image', img)
cv2.waitKey()

kp, hQ = fe.imageSKL(img_path, kmeans, flag = 'so')
print 'hQ are as follows: \n %s \n' % hQ
print 'There are %d words in the test image. \n' % np.sum(hQ)

# image_idx: list of length 3 that contains indexes of the matched images
# max_sim: list of length 3 that contains the similarity scores of the matched images
# similarity scores are descending
image_idx, max_sim = fe.findObject(vector, hQ, dis_measure = 'NHI')
print '\n The images most similar to the test image: %s.' % image_idx
print 'Similarity score: %s.' % max_sim

# Generate pool of path to candidate images
path_pool = ['.\\Groundhog day\\' + 'I_{:05d}.jpg'.format(x) for x in xrange(60, 3000, 30)]

for idx in xrange(3):
    match_img = cv2.imread(path_pool[image_idx[idx]])
    cv2.namedWindow('match')
    cv2.imshow('match', match_img)
    cv2.waitKey()

cv2.destroyAllWindows()
