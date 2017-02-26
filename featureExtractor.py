import numpy as np
import cv2
import collections
from sklearn.cluster import KMeans

def hcSift(img_name, show = False):
    # Extract local interest points with the Harris Corner Detector
    # and describe the extracted points by SIFT descriptor
    # img_name: string that stores relative of absolute path to input image
    # img_name eg '.\\Groundhog day\\I_00060.jpg'
    pass


def msSift(img_name, show = False):
    # Extract local interest points with the MSER detector
    # and describe the extracted points by SIFT descriptor
    # img_name: string that stores relative of absolute path to input image
    # img_name eg '.\\Groundhog day\\I_00060.jpg'

    img = cv2.imread(img_name)
    if show:
        cv2.namedWindow('img')
        cv2.imshow('img', img)
        cv2.waitKey(200)
        cv2.destroyWindow('img')
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # mser_region is a list
    # its elements are nparrays containing coordinates of it pixels
    mser = cv2.MSER()
    fd = cv2.FeatureDetector_create('MSER')
    # kpts is a list
    kp = fd.detect(gray)

    sift = cv2.SIFT()
    # des is a numpy array of shape (num_of_kp, 128)
    # compute SIFT features at places determined by kp
    kp, des = sift.compute(gray, kp)

    return gray, kp, des


def siftOnly(img_name, show = False):
    # Extract SIFT descriptor of the input image
    # img_name: string that stores relative of absolute path to input image
    # img_name eg '.\\Groundhog day\\I_00060.jpg'

    img = cv2.imread(img_name)
    if show:
        cv2.namedWindow('img')
        cv2.imshow('img', img)
        cv2.waitKey(200)
        cv2.destroyWindow('img')
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    sift = cv2.SIFT()
    # des is a numpy array of shape (num_of_kp, 128)
    # compute SIFT features at places determined by kp
    kp, des = sift.detectAndCompute(gray, None)

    return gray, kp, des


def loadData(path, start = 60, stop = 999, step = 60, flag = 'ms'):

    img_pool = [path + 'I_{:05d}.jpg'.format(x) for x in xrange(start, stop, step)]

    if flag == 'ms':
        gray, kp, des = msSift(img_pool[0])
    elif flag == 'so':
        gray, kp, des = siftOnly(img_pool[0])
    kp = [kp]

    for idx, img_name in enumerate(img_pool[1:]):
        if flag == 'ms':
            img_data = msSift(img_name)
        elif flag == 'so':
            img_data = siftOnly(img_name)

        gray = np.dstack((gray, img_data[0]))
        if (img_data[1] != None) and (img_data[2] != None):
            kp.append(img_data[1])
            des = np.vstack((des, img_data[2]))
        else:
            kp.append([])
            des = np.vstack((des, np.zeros(128)))


        print 'the %d-th imgae, descriptor shape is: %s \n' % (idx + 1, des.shape)

    return img_pool, gray, kp, des


def cluster(des, num_cluster):

    # iter: stores max number of iteration performed by the kmeans algorithm
    # attempt: stores number of attempts of the kmeans algorithm
    iter = 1000
    attempt = 30

    # Data type conversion, function cv2.kmeans only accepts float as input
    des = np.float32(des)
    criteria = (cv2.TERM_CRITERIA_MAX_ITER+cv2.TERM_CRITERIA_EPS, iter, 0.001)
    # Use kmean++ for better performance
    # flag = cv2.KMEANS_PP_CENTERS
    flag = cv2.KMEANS_RANDOM_CENTERS

    compact, best_label, center = cv2.kmeans(des, num_cluster, criteria, attempt, flag)
    print 'The features are now clustered into %d clusters.' % num_cluster

    return best_label, center


def clusterSKLearn(des, num_cluster, eps = 1e-4):
    kmeans = KMeans(num_cluster, max_iter = 500, tol = eps).fit(des)
    return kmeans


def wordGeneration(kp, des, V, label):
    # Represent each input images with a vector (visual phrase)
    # V: integer, number of clusters

    # I: integer, number of images
    I = len(kp)

    cur = 0
    # Original label is of shape (, 1)
    # Flattened label is of shape (,)
    label = label.flatten()

    # Each row of vector represents an image
    vector = np.zeros((I, V))
    for img_num in xrange(I):
        c = collections.Counter(label[cur: cur+len(kp[img_num])])
        for key, occurrence in c.iteritems():
            vector[img_num, int(key)] = occurrence
        print 'The %d-th image is represented as: %s' % (img_num + 1, vector[img_num, :])

    # Store number of documents that a specific word occurs
    # doc_occ[i]: number of documents that the i-th word occurs
    doc_occ = np.sum(vector > 0, axis = 0)
    print 'Number of documents (images) that a specific word occurs: %s' % doc_occ

    return vector.astype(float), doc_occ.astype(float)


def stopList(vector, k = 0.05):
    # k: float, stores percentage of words that appears in the stop list

    # V: integer, number of clusters equals to number of words
    V = vector.shape[1]
    print V

    occurrence = np.sum(vector, axis = 0)
    list = []

    for i in xrange(int(k * V)):
        id_max = np.argmax(occurrence)
        list.append(id_max)
        print 'The %d-th most frequent word appears %d times.' % (i + 1, occurrence[id_max])
        occurrence[id_max] = 0

    return list


def imagePhrase(img_name, center, flag = 'ms'):
    # Read test image and represent it as a visual phrase

    if flag == 'ms':
        gray, kp, des = msSift(img_name, show = False)
    elif flag == 'so':
        gray, kp, des = siftOnly(img_name, show = False)

    # Newer and quicker version of the code below
    phrase = np.zeros(center.shape[0])
    for idx, descriptor in enumerate(des[:]):
        distance = np.sum((descriptor - center) ** 2, axis = 1)
        phrase[np.argmin(distance)] += 1

    return kp, phrase.astype(float)

#    label = np.zeros(des.shape[0])
#    for idx, descriptor in enumerate(des[:]):
#        distance = np.sum((descriptor - center) ** 2, axis = 1)
#        # print 'Distance for %d-th descriptor: %s. \n' % (idx + 1, distance)
#        label[idx] = np.argmin(distance)
#        # print 'The %d-th descriptor is converted to word %d. \n' % (idx + 1, label[idx])#

#    print 'The labels for all the descriptors: %s. \n' % label#

#    phrase = [0 for i in xrange(center.shape[0])]
#    c = collections.Counter(label)
#    for key, occurrence in c.iteritems():
#        phrase[int(key)] = occurrence
#
#    return kp, phrase


def imageSKL(img_name, kmeans, flag = 'ms'):
    # Read test image and represent it as a visual phrase

    if flag == 'ms':
        gray, kp, des = msSift(img_name, show = False)
    elif flag == 'so':
        gray, kp, des = siftOnly(img_name, show = False)

    center = kmeans.cluster_centers_
    # Newer and quicker version of the code below
    phrase = np.zeros(center.shape[0])
    labels = kmeans.predict(des)
    for label in labels:
        phrase[label] += 1

    return kp, phrase.astype(float)


def findObject(vector, phrase, dis_measure = 'NHI', top = 3):

    # print 'Input values: \n', phrase, '\n', vector

    sim = np.sum(np.minimum(vector, phrase), axis = 1) / (np.sum(np.maximum(vector, phrase), axis = 1) + 1e-5)

    print '\n \n Similarity: \n', sim, '\n \n'
    c_sim = np.copy(sim)

    image_idx = []
    max_sim = []
    for t in xrange(top):
        image_idx.append(np.argmax(c_sim))
        max_sim.append(np.max(c_sim))
        c_sim[np.argmax(c_sim)] = 0

    return image_idx, max_sim
