'''
import cv2
import matplotlib.pyplot as plt

# reading a image
#img = cv2.imread('2000.jpg')
#img = cv2.imread('2000.jpg')
img = cv2.imread('2000.jpg', cv2.IMREAD_UNCHANGED)
print('Original Dimensions : ',img.shape)
scale_percent = 60
width = int(img.shape[1] * scale_percent / 100)
height = int(img.shape[0] * scale_percent / 100)
dim = (width, height)
# resize image
resized = cv2.resize(img, dim, interpolation = cv2.INTER_AREA)
imgGrey = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
imgCanny = cv2.Canny(img, 100, 100)
orb = cv2.ORB_create()  # OpenCV 3 backward incompatibility: Do not create a detector with `cv2.ORB()`.
key_points, description = orb.detectAndCompute(imgCanny, None)
img_building_keypoints = cv2.drawKeypoints(imgCanny,
                                           key_points,
                                           imgCanny,
                                           flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)  # Draw circles.
plt.figure(figsize=(16, 16))
plt.title('ORB Interest Points')
plt.imshow(img_building_keypoints)
plt.show()


def image_detect_and_compute(detector, img_name):
    img = cv2.imread(img_name)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    kp, des = detector.detectAndCompute(img, None)
    return img, kp, des


def draw_image_matches(detector, img1_name, img2_name, nmatches=10):
    img1, kp1, des1 = image_detect_and_compute(detector, img1_name)
    img2, kp2, des2 = image_detect_and_compute(detector, img2_name)

    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = bf.match(des1, des2)
    matches = sorted(matches, key=lambda x: x.distance)  # Sort matches by distance.  Best come first.

    img_matches = cv2.drawMatches(img1, kp1, img2, kp2, matches[:nmatches], img2, flags=2)  # Show top 10 matches
    plt.figure(figsize=(16, 16))
    plt.title(type(detector))
    plt.imshow(img_matches)
    plt.show()


orb = cv2.ORB_create()
draw_image_matches(orb, '2000.jpg', '200f.jpg')
'''
import cv2
import numpy as np
# import matplotlib.pyplot as plt
import os

# print_list = []

# function to print string to image
'''
def print_f(sen):
    global print_list
    print_list = print_list + [sen]

# function to show printed string
def print_s(_header, DefaultCanvas=None):
    if DefaultCanvas !=  None :
        print_list = DefaultCanvas
    else:
        global print_list
    x = len(print_list)
    canvas = np.ones((20*x+10,512))*255
    font = cv2.FONT_HERSHEY_COMPLEX_SMALL
    for i in range(0,x):
        cv2.putText(canvas,print_list[i] ,(10,(i+1)*20), font, 1,(0,255,0),1,cv2.LINE_AA)
    cv2.imshow(_header,canvas)
'''


# get file list from folder
def getFiles(_root):
    return next(os.walk(_root))[2]


# take image input
def takeImageInput(_root, _path):
    _path = _root + _path
    colored = cv2.imread(_path)
    _grayscale = cv2.cvtColor(colored, cv2.COLOR_BGR2GRAY)
    colored = cv2.cvtColor(colored, cv2.COLOR_BGR2RGB)
    return (colored, _grayscale)


# image preprocessing and noise reduction
def initialTransformations(_image):
    global thresh, blur, canny, filters
    _image = cv2.bilateralFilter(_image, 9, 50, 50)  # preserve edge and remove noise
    filters = _image
    _image = cv2.GaussianBlur(_image, (11, 11), 4)  #
    blur = _image
    _thresh = cv2.adaptiveThreshold(_image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11,
                                    2)  # binary threshold
    _thresh = cv2.medianBlur(_thresh, 11)  #
    thresh = _thresh
    _image = _thresh
    # _image = cv2.Canny(_image,5,10)
    canny = _image
    return _image


# returns set of points to be enclosed by bounding rectangle
def maxContour(_contour):
    global imageArea
    _pointList = np.array(_contour[0][:, 0])  # list of list
    _maxContourArea = 0
    for cnt in _contour:
        area = cv2.contourArea(cnt)
        if area / imageArea < 0.9:
            _pointList = np.append(_pointList, cnt[:, 0], 0)
    return _pointList


def dist(_x, _y):
    return ((_x[0] - _y[0]) ** 2 + (_x[1] - _y[1]) ** 2) ** 0.5


def getFit(_image, box, clip=None):
    br = box[0]
    bl = box[1]
    tl = box[2]
    tr = box[3]

    _width = dist(br, bl)
    _height = dist(bl, tl)

    _pts1 = np.float32([br, bl, tl, tr])
    _pts2 = np.float32([[_width, _height], [0, _height], [0, 0], [_width, 0]])

    transformationMatrix = cv2.getPerspectiveTransform(_pts1, _pts2)
    transImage = cv2.warpPerspective(_image, transformationMatrix, (int(_width), int(_height)))
    if (clip != None):
        transImage = transImage[clip[1]:int(_height) - clip[3], clip[0]: int(_width) - clip[2]]
    if (transImage.shape[0] > transImage.shape[1]):
        M = cv2.getRotationMatrix2D((transImage.shape[0] / 2, transImage.shape[0] / 2), -90, 1)
        transImage = cv2.warpAffine(transImage, M, (transImage.shape[0], transImage.shape[1]))
    return transImage


# match two images and generate descriptors
def imageMatcher(_standardImage, _sampleImage):
    _orb = cv2.ORB_create()
    try:
        _keyPoint_1, _descriptors_1 = _orb.detectAndCompute(_standardImage, None)
        _keyPoint_2, _descriptors_2 = _orb.detectAndCompute(_sampleImage, None)
        # In case there is no keypoint in sample image
        if len(_keyPoint_1) == 0 or len(_keyPoint_2) == 0:
            return (None, None)
        # Create a brute force matcher
        bruteForceMatcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
        # Matches descriptors of images
        _matches = bruteForceMatcher.match(_descriptors_1, _descriptors_2)
        # sort matches wrt distance: less distance => more accurate match
        _matches = sorted(_matches, key=lambda x: x.distance)
        return (_matches, (_keyPoint_1, _keyPoint_2))
    except Exception as e:
        # print(e.args)
        return (None, (None, None))


# detemines accuracy points on the matches
def determineAccuracy(_matches, _limit=None):
    if _limit == None or _limit == 0:
        _limit = 3 / 2
    _sum = 0
    _limit = int(len(_matches) / _limit)
    for _ in _matches[:_limit]:
        _sum += _.distance
    _avg = float(0)
    try:
        _avg = _sum / _limit
    except Exception as e:
        # print(e.args)
        return int(1e10)
    return _avg


# draw matches to a new image
def drawMatcher(_standardImage, _sampleImage, _keyPoints, _matches):
    result = None
    result = cv2.drawMatches(_standardImage, _keyPoints[0], _sampleImage,
                             _keyPoints[1], _matches, result, flags=2)
    return result


def getFeatureImage(_fullImage, _feature):
    _height = _fullImage.shape[0]
    _width = _fullImage.shape[1]
    _x = int(_height * _feature[0])
    _y = int(_width * _feature[1])
    _dx = int(_height * _feature[2])
    _dy = int(_width * _feature[3])

    return _fullImage[_x:_x + _dx, _y: _y + _dy]


# ----------STARTS---HERE------------------------#

# -----ROOT - folder containing all the test note images-------------
# -----Pass root to getFiles() to get list of files in that folder---
# ----------DATA SET STARTS------------

notes = ['10', '20', '50', '100', '200', '500']
artio = [2.075723367, 2.212180173, 1.9959253487, 2.10026738, 2.1148898655, 2.16519103]
tolerance = [0.004586393, 0.006160723, 0.008263391, 0.024120747, 0.019034105, 0.020002683]
feature_set = [
    [0.2331274, 0.0096001, 0.2236, 0.0644],
    [0.243820225, 0.947776629, 0.235955056, 0.051706308],
    [0.08688764, 0.097207859, 0.120224719, 0.257497415],
    [0.100482759, 0.577044025, 0.099827586, 0.32468535],
    [0.401345291, 0.167958656, 0.293721973, 0.069767442],
    [0.66313, 0.7095218, 0.1455, 0.17838],
    [0.495689655, 0.29009434, 0.132758621, 0.027122642],
    [0.18362069, 0.543632075, 0.735344828, 0.037735849],
    [0.570786517, 0.88262668, 0.265168539, 0.084281282],
]

feature_set_old = [
    [0.324112769, 0.385028302, 0.2039801, 0.2044009434],
    [0.247159451, 0.843353597, 0.161372756, 0.111560284],
    [0.055966209, 0.072921986, 0.173522703, 0.133981763],
    [0.067581837, 0.322695035, 0.1731151, 0.339148936],
    [0.610538543, 0.024822695, 0.299271383, 0.086119554],
    [0.648363252, 0.873353597, 0.173178458, 0.088652482]
]

feature_list = ['L_BRAILLE', 'R_BRAILLE', 'RBI_HI', 'RBI_EN', 'VALUE_STD', 'VALUE_HI', 'VALUE_HID', 'SEC_STRIP',
                'EMBLEM']
feature_list_old = ['VALUE_CENTER', 'VALUE_RIGHT', 'VALUE_LEFT', 'RBI_EN_HI', 'EMBLEM', 'SEAL']
# -----------DATA SET ENDS-------------

root = "../Image/new_m/"
fileList = getFiles(root)  # return file list
fileList = sorted(fileList)
for i in range(23, 34):
    path = fileList[i]

    # print(path)
    coloredImage, grayscale = takeImageInput(root, path)

    height = coloredImage.shape[0]
    width = coloredImage.shape[1]
    imageArea = float(height * width)

    # prepares image for contour detectection
    image = initialTransformations(grayscale)

    img, contours, hierarchy = cv2.findContours(image, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

    # canvas = np.ones((height, width, 3))

    pointList = maxContour(contours)
    rect = cv2.minAreaRect(pointList)
    rect = cv2.boxPoints(rect)
    rect = np.int0(rect)

    proper_image = getFit(coloredImage, rect)
    # ============ PROPER IMAGE TO PROCESS =================
    # referenceImage = cv2.imread('../../../Image/500/0005.jpg',0)
    # sample = getFeatureImage(proper_image, feature_set[4])

    detect_feat = [0, 1, 4, 5]
    verify_feat = [2, 3, 7, 8, 9]

    detect_feat_old = [0, 1, 2]
    verify_feat_old = [3, 4, 5]

    notes_type_old = ['10', '20', '50', '100', 'Undetected']
    notes_type = ['500', '200', 'Undetected']

    winner_old = [0, 0, 0, 0, 0]
    winner = [0, 0, 0]
    lead = 0

    feat_acc = np.zeros(detect_feat)
    feat_acc_old = np.zeros(detect_feat_old)
    # ========================= TYPE DETECTION ================================================
    for x_out, d_feat in enumerate(detect_feat):
        mini = 100  # hamming distance min
        lead = 0  # leader matching
        for i_note in range(0, len(notes_type) - 1):
            # print('NOTE : ' + notes_type[i_note] + ', FEATURE: ' + feature_list[d_feat])

            sample = getFeatureImage(proper_image, feature_set[d_feat])

            reference_image = cv2.imread('/Image/' + notes_type[i_note] + '/000' + str(d_feat + 1) + '.jpg', 0)
            try:
                matches, kp = imageMatcher(reference_image, sample)
                if matches == None or kp == None:
                    # print('Feature ' + feature_list[d_feat] + ' failed.')
                    continue
                accuracy = determineAccuracy(matches)
                if mini > accuracy:
                    lead = i_note
                    mini = accuracy
                # mat_image = drawMatcher(reference_image, sample, kp, matches[0:20])
                # print('Feature ' + feature_list[d_feat] + ' matches with accuracy  : ' + str(accuracy))
            except:
                pass
                # print('Error in feature ' + feature_list[d_feat] + ' of note ' + str(i))
        if mini > 60:
            lead = 2
        winner[lead] += 1
    lead = winner.index(max(winner))
    if winner[2] == max(winner):
        lead = 2

    if lead != 2:
        print(winner)
        print(str(i) + ' - Detected Note : ' + notes_type[lead])
        title = str(i) + " " + notes_type[lead]
        cv2.namedWindow(title, cv2.WINDOW_KEEPRATIO)
        cv2.imshow(title, proper_image)
    # ======================= FOR OLD NOTES ONLY WHEN UNDETECTED ======================
    if lead == 2:
        for x_out, d_feat in enumerate(detect_feat_old):
            mini = 100
            lead = 0
            for i_note in range(0, len(notes_type_old) - 1):
                # print('NOTE : ' + notes_type_old[i_note] + ', FEATURE: ' + feature_list_old[d_feat])

                sample = getFeatureImage(proper_image, feature_set_old[d_feat])
                reference_image = cv2.imread(
                    '../Image/' + notes_type_old[i_note] + '/000' + str(d_feat + 1) + '.jpg', 0)
                try:
                    matches, kp = imageMatcher(reference_image, sample)
                    if matches == None or kp == None:
                        # print('Feature ' + feature_list_old[d_feat] + ' failed.')
                        continue
                    accuracy = determineAccuracy(matches)
                    if mini > accuracy:
                        lead = i_note
                        mini = accuracy
                    mat_image = drawMatcher(reference_image, sample, kp, matches[0:20])
                    # title = str(d_feat) + " "  + notes_type_old[i_note]
                    # cv2.namedWindow(title, cv2.WINDOW_KEEPRATIO)
                    # cv2.imshow(title, mat_image)
                    # print('Feature ' + feature_list_old[d_feat] + ' matches with accuracy  : ' + str(accuracy))
                except:
                    pass
                    # print('Error in feature ' + feature_list[d_feat] + ' of note ' + str(i))
            if mini > 64:
                lead = 4
            winner_old[lead] += 1
        lead = winner_old.index(max(winner_old))
        if winner_old[4] == max(winner_old):
            lead = 4

        print(winner_old)
        print(str(i) + ' - Detected Note : ' + notes_type_old[lead])
        title = str(i) + " " + notes_type_old[lead]
        cv2.namedWindow(title, cv2.WINDOW_KEEPRATIO)
        cv2.imshow(title, proper_image)

    # =======================  VERIFICATION ===========================================
    # for iter, feature_x in enumerate(feature_set):
    #     sample = getFeatureImage(proper_image, feature_x)
    #     reference_image = cv2.imread('../../../Image/500/000' + str(iter + 1) + '.jpg', 0)
    #     try:
    #         matches , kp = imageMatcher(reference_image, sample)
    #         if matches == None or kp == None:
    #             print('Feature ' + feature_list[iter] + ' failed.')
    #             continue
    #         accuracy = determineAccuracy(matches)
    #         mat_image = drawMatcher(reference_image, sample, kp, matches[0:20])
    #         resultant_image = mat_image
    #         title = str(iter)
    #         cv2.namedWindow(title, cv2.WINDOW_KEEPRATIO)
    #         cv2.imshow(title, resultant_image)
    #         print('Feature ' + feature_list[iter] + ' matches with accuracy  : ' + str(accuracy))
    #     except:
    #         print('Error in feature 'c + feature_list[iter] + ' of note ' + str(i))
    # ============================= OLD NOTES VERIFICATION ==========================================
    acMeasure = np.ones(len(verify_feat_old))
    threshold = np.array([65, 65, 65])
    for iter, feature_x in enumerate(verify_feat_old):
        sample = getFeatureImage(proper_image, feature_set_old[feature_x])
        reference_image = cv2.imread('../Image/' + notes_type_old[lead] + '/000' + str(feature_x + 1) + '.jpg', 0)
        try:
            matches, kp = imageMatcher(reference_image, sample)
            if matches == None or kp == None:
                print('Feature ' + feature_list_old[feature_x] + ' failed.')
                continue
            accuracy = determineAccuracy(matches)
            acMeasure[iter] = (1 - accuracy / threshold[iter])

            mat_image = drawMatcher(reference_image, sample, kp, matches[0:20])
            resultant_image = mat_image
            title = str(iter)
            cv2.namedWindow(title, cv2.WINDOW_KEEPRATIO)
            cv2.imshow(title, resultant_image)
            print('Feature ' + feature_list_old[feature_x] + ' matches with distance value  : ' + str(accuracy))
        except:
            print('Error in feature ' + feature_list_old[feature_x] + ' of note ' + str(i))
    print("ACCURACY : " + str(1 - np.sum(acMeasure * (1 - np.array([0.5, 0.2, 0.3])))))

    # ===================================================================================== #
    # try:
    #     matches , keyPoints = imageMatcher(referenceImage, sample)
    #     if matches == None or keyPoints == None:
    #         print(str() + 'No KeyPoints Found.')
    #         cv2.imshow("Failed Imqage : " + str(i) , sample)
    #         continue
    #     mat_image = drawMatcher(referenceImage, sample, keyPoints, matches[0:20])
    #     resultant_image = mat_image
    #     title = str(determineAccuracy(matches))
    #     cv2.namedWindow(title, cv2.WINDOW_KEEPRATIO)
    #     cv2.imshow(title, resultant_image)
    # except:
    #     print("Error in " + str(i))
    if cv2.waitKey(0) == ord('c'):
        break

cv2.waitKey(0)






