import numpy as np
import cv2
import pickle
import featureExtractor as fe

drawing = False # True if mouse is pressed
corner = [] # Store coordinates of two opposite corners of the ROI
img = None

# Mouse callback function
def mouseClick(event, x, y, flags, param):
    global drawing, corner, img

    if event == cv2.EVENT_LBUTTONDOWN:
        drawing = True
        corner = [(x, y)]

    elif event == cv2.EVENT_LBUTTONUP:
        drawing = False
        corner.append((x, y))
        cv2.rectangle(img, corner[0], corner[1], (0,255,0), 2)


# Due to problems about global variable img,
# this function could not be used
def selectRoi(img_path):

    ori_img = cv2.imread(img_path)
    img = cv2.cvtColor(ori_img, cv2.COLOR_BGR2GRAY)
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
        roi = np.copy(ori_img[min_x:max_x, min_y:max_y])

    cv2.namedWindow('roi')
    cv2.imshow('roi', roi)
    if cv2.waitKey() == 27:
        cv2.destroyAllWindows()

    return roi

#
#
def partition(img, M, N):

    row = img.shape[0]
    col = img.shape[1]
    assert row > 10*M and col > 10*N, 'Too many partitions!'

    x = np.random.randint(2, row-2, M-1)
    y = np.random.randint(2, col-2, N-1)
    x.sort()
    x = np.hstack((0, x, row))
    y.sort()
    y = np.hstack((0, y, col))

    sub_list = []
    for idx in xrange(x.shape[0]-1):
        for idy in xrange(y.shape[0]-1):
            sub_img = np.copy(img[x[idx]:x[idx+1], y[idy]:y[idy+1]])
            sub_list.append((sub_img, (x[idx], y[idy]), (x[idx+1], y[idy+1])))

    return sub_list


def pixelScore(sub, kmeans, list, hQ, img_shape):
    # img_shape is a tuple (width, height)

    center = kmeans.cluster_centers_
    hP = np.zeros((len(sub), center.shape[0]), dtype = float)
    score = np.zeros(img_shape)

    for idx, (sub_img, topleft, botright) in enumerate(sub):
        sift = cv2.SIFT()
        if (sub_img.shape[0] != 0) and (sub_img.shape[1] != 0):
            kp, des = sift.detectAndCompute(sub_img, None)
            if des != None:
                labels = kmeans.predict(des)
                for label in labels:
                    hP[idx, label] += 1
                hP[idx, list] = 0

        sim = np.sum(np.minimum(hP[idx], hQ)) / (np.sum(np.maximum(hP[idx], hQ)) + 1e-6)
        score[topleft[0]:botright[0], topleft[1]:botright[1]] = sim

    return hP, score



# Test code for locateObj.py
# Test whether RVP algorithm works as expected

img_path = '.\\Groundhog day\\I_02000.jpg'
match_path = '.\\Groundhog day\\I_02001.jpg'

kmeans_filename = '.\\Model\\kmeans'
stoplist_filename = '.\\Model\\stoplist'
score_filename = '.\\Model\\score'

# M: integer, number of rows in each partition, eg 8
# N: integer, number of columns in each partition, eg 16
# K: integer, total number of partition, eg 200
# alpha: float, coefficient used to conpute threshold for object localization
(M, N, K) = (8, 8, 50)
alpha = 2.6

# Read image that contains roi (region of interest)
# and select roi
img = cv2.imread(img_path)
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
cv2.namedWindow('image')
cv2.setMouseCallback('image', mouseClick)#
while True:
    cv2.imshow('image', img)
    key = cv2.waitKey(1) & 0xFF
    if key == 27:
        break

cv2.destroyAllWindows()
cv2.imwrite('.//Results//original_img.jpg', img)

# Crop the roi out
if len(corner) == 2:
    min_x = min(corner[0][1], corner[1][1])
    max_x = max(corner[0][1], corner[1][1])
    min_y = min(corner[0][0], corner[1][0])
    max_y = max(corner[0][0], corner[1][0])
    roi = np.copy(gray[min_x:max_x, min_y:max_y])

cv2.namedWindow('roi')
cv2.imshow('roi', roi)
cv2.waitKey(2000)
cv2.imwrite('.//Results//roi.jpg', roi)

# Extract SIFT descriptors of roi
sift = cv2.SIFT()
kp, des = sift.detectAndCompute(roi, None)


# Load the pre-computed bag of words, kmeans object and stop list
with open(kmeans_filename, 'rb') as kmeans_in:
    kmeans = pickle.load(kmeans_in)
with open(stoplist_filename, 'rb') as stoplist_in:
    stoplist = pickle.load(stoplist_in)


center = kmeans.cluster_centers_
# hQ: visual phrase that represent roi
hQ = np.zeros(center.shape[0])
labels = kmeans.predict(des)
for label in labels:
    hQ[label] += 1
hQ[stoplist] = 0

del label, labels
print 'hQ are as follows: \n %s \n' % hQ#


# Read image that we are going to search for roi
img = cv2.imread(match_path)
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
# sub: list of length M*N,
# each element of sub: (sub-image, top-left corner, bottom-right corner)
sub = partition(gray, M, N)
# Compute similarity score for each pixel
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
cv2.waitKey(2000)
cv2.imwrite('.//Results//score.jpg', show_score * 255)

# Store original similarity score array
with open(score_filename, 'wb') as score_out:
    pickle.dump(score, score_out)

# Compute threshold for object localization
thres = alpha * np.sum(show_score) / (gray.shape[0] * gray.shape[1])
print 'The threshold is: %f' % thres
mask = np.zeros(gray.shape)
mask[show_score >= thres] = 1

cv2.namedWindow('match')
cv2.imshow('match', img)
cv2.waitKey(2000)


img = img.astype(float)
img /= 255
img *= mask[:, :, None]
cv2.namedWindow('image')
cv2.imshow('image', img)
cv2.imwrite('.//Results//final_result.jpg', img * 255)

cv2.waitKey(5000)
cv2.destroyAllWindows()

