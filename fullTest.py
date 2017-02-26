# Test the whole system
# Copy the code into locateOBj.py

import numpy as np
import cv2
import pickle
import featureExtractor as fe

kmeans_filename = '.\\data\\kmeans'
vector_filename = '.\\data\\vector'
doc_filename = '.\\data\\doc_occ'
stoplist_filename = '.\\data\\stoplist'
score_filename = '.\\data\\score'

img_path = '.\\Groundhog day\\I_02276.jpg'
cluster_num = 200
(M, N, K) = (8, 8, 50)
alpha = 2.6

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
cv2.setMouseCallback('image', mouseClick)
while True:
    cv2.imshow('image', img)
    key = cv2.waitKey(1) & 0xFF
    if key == 27:
        break

cv2.destroyAllWindows()

if len(corner) == 2:
    min_x = min(corner[0][1], corner[1][1])
    max_x = max(corner[0][1], corner[1][1])
    min_y = min(corner[0][0], corner[1][0])
    max_y = max(corner[0][0], corner[1][0])
    roi = np.copy(gray[min_x:max_x, min_y:max_y])#

cv2.namedWindow('roi')
cv2.imshow('roi', roi)
cv2.waitKey()

sift = cv2.SIFT()
kp, des = sift.detectAndCompute(roi, None)

hQ = np.zeros(center.shape[0])
labels = kmeans.predict(des)
for label in labels:
    hQ[label] += 1
hQ[stoplist] = 0

del label, labels
print 'hQ are as follows: \n %s \n' % hQ
print 'There are %d words in the test image. \n' % np.sum(hQ)

# kp, hQ = fe.imageSKL(img_path, kmeans, flag = 'so')

image_idx, max_sim = fe.findObject(vector, hQ, dis_measure = 'NHI')
print '\n The images most similar to the test image: %s.' % image_idx
print 'Similarity score: %s.' % max_sim

path_pool = ['.\\Groundhog day\\' + 'I_{:05d}.jpg'.format(x) for x in xrange(60, 3000, 30)]

match_img = cv2.imread(path_pool[image_idx[0]])
cv2.namedWindow('match')
cv2.imshow('match', match_img)
cv2.waitKey()
cv2.destroyWindow('match')


gray = cv2.cvtColor(match_img, cv2.COLOR_BGR2GRAY)
sub = partition(gray, M, N)
hP, score = pixelScore(sub, kmeans, stoplist, hQ, gray.shape)

for k in xrange(K - 1):
    sub = partition(gray, M, N)
    hP, score_k = pixelScore(sub, kmeans, stoplist, hQ, gray.shape)
    score = score + score_k#

show_score = np.copy(score)
print 'show_score data type: ', show_score.dtype, '\n'
# Normalize show_score to range [0, 1]
show_score /= np.max(show_score)#
cv2.namedWindow('score')
cv2.imshow('score', show_score)
cv2.waitKey()

with open(score_filename, 'wb') as score_out:
    pickle.dump(score, score_out)

thres = alpha * np.sum(show_score) / (gray.shape[0] * gray.shape[1])
print 'The threshold is: %f' % thres
cv2.namedWindow('thres')
mask = np.zeros(gray.shape)
mask[show_score >= thres] = 1

match_img = match_img.astype(float)
match_img /= 255
match_img *= mask[:, :, None]
cv2.namedWindow('image')
cv2.imshow('image', match_img)

key = cv2.waitKey() & 0xFF
if key == 27:
    cv2.destroyAllWindows()
