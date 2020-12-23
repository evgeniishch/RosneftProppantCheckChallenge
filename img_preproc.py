from skimage.filters import threshold_local
import numpy as np
import pandas as pd
import cv2
from sklearn.multioutput import MultiOutputRegressor
from sklearn.tree import DecisionTreeRegressor

def resize(image, width=None, height=None, inter=cv2.INTER_AREA):
    # initialize the dimensions of the image to be resized and
    # grab the image size
    dim = None
    (h, w) = image.shape[:2]

    # if both the width and height are None, then return the
    # original image
    if width is None and height is None:
        return image

    # check to see if the width is None
    if width is None:
        # calculate the ratio of the height and construct the
        # dimensions
        r = height / float(h)
        dim = (int(w * r), height)

    # otherwise, the height is None
    else:
        # calculate the ratio of the width and construct the
        # dimensions
        r = width / float(w)
        dim = (width, int(h * r))

    # resize the image
    resized = cv2.resize(image, dim, interpolation=inter)

    # return the resized image
    return resized

def grab_contours(cnts):
    # if the length the contours tuple returned by cv2.findContours
    # is '2' then we are using either OpenCV v2.4, v4-beta, or
    # v4-official
    if len(cnts) == 2:
        cnts = cnts[0]

    # if the length of the contours tuple is '3' then we are using
    # either OpenCV v3, v4-pre, or v4-alpha
    elif len(cnts) == 3:
        cnts = cnts[1]

    # otherwise OpenCV has changed their cv2.findContours return
    # signature yet again and I have no idea WTH is going on
    else:
        raise Exception(("Contours tuple must have length 2 or 3, "
            "otherwise OpenCV changed their cv2.findContours return "
            "signature yet again. Refer to OpenCV's documentation "
            "in that case"))

    # return the actual contours array
    return cnts



def rotate_image(mat, angle):
    """
    Rotates an image (angle in degrees) and expands image to avoid cropping
    """

    height, width = mat.shape[:2] # image shape has 3 dimensions
    image_center = (width/2, height/2) # getRotationMatrix2D needs coordinates in reverse order (width, height) compared to shape

    rotation_mat = cv2.getRotationMatrix2D(image_center, angle, 1.)

    # rotation calculates the cos and sin, taking absolutes of those.
    abs_cos = abs(rotation_mat[0,0]) 
    abs_sin = abs(rotation_mat[0,1])

    # find the new width and height bounds
    bound_w = int(height * abs_sin + width * abs_cos)
    bound_h = int(height * abs_cos + width * abs_sin)

    # subtract old image center (bringing image back to origo) and adding the new image center coordinates
    rotation_mat[0, 2] += bound_w/2 - image_center[0]
    rotation_mat[1, 2] += bound_h/2 - image_center[1]

    # rotate image with the new bounds and translated rotation matrix
    rotated_mat = cv2.warpAffine(mat, rotation_mat, (bound_w, bound_h))
    return rotated_mat


def order_points(pts):
	rect = np.zeros((4, 2), dtype = "float32")
	s = pts.sum(axis = 1)

	rect[0] = pts[np.argmin(s)]
	rect[2] = pts[np.argmax(s)]

	diff = np.diff(pts, axis = 1)
	rect[1] = pts[np.argmin(diff)]
	rect[3] = pts[np.argmax(diff)]
	return rect


def four_point_transform(image, pts):
	rect = order_points(pts)
	(tl, tr, br, bl) = rect

	widthA = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
	widthB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
	maxWidth = max(int(widthA), int(widthB))

	heightA = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
	heightB = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
	maxHeight = max(int(heightA), int(heightB))

	dst = np.array([
		[0, 0],
		[maxWidth - 1, 0],
		[maxWidth - 1, maxHeight - 1],
		[0, maxHeight - 1]], dtype = "float32")

	M = cv2.getPerspectiveTransform(rect, dst)
	warped = cv2.warpPerspective(image, M, (maxWidth, maxHeight))

	return warped


def planarize(image):
    ratio = image.shape[0] / 500.0
    orig = image.copy()
    image = resize(image, height = 500)

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (5, 5), 0)
    edged = cv2.Canny(gray, 75, 200)

    cnts = cv2.findContours(edged.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    cnts = grab_contours(cnts)
    cnts = sorted(cnts, key = cv2.contourArea, reverse = True)[:5]

    for c in cnts[2:]:
        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.02 * peri, True)

        if len(approx) == 4:
            screenCnt = approx
            break

    warped = four_point_transform(orig, screenCnt.reshape(4, 2) * ratio)

    h, w, _ = warped.shape

    if w < h:
        dim = (1200, 2000)
    else:
        dim = (2000, 1200)
    # resize image
    warped = cv2.resize(warped, dim, interpolation = cv2.INTER_AREA)

    if w < h:
        warped = rotate_image(warped, 90)

    return warped


def getCountHoughCircles(img, param_list):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    circles = cv2.HoughCircles(img, 
                               cv2.HOUGH_GRADIENT, 
                               param_list[0] if param_list[0] > 0 else 2, 
                               round(param_list[1]) if param_list[1] > 0 else 5,
                               param1=param_list[2],
                               param2=param_list[3],
                               minRadius=5, 
                               maxRadius=9)
    if circles is not None:
        return circles.shape[1]
    else:
        return None


def mean_color(image):
    return image.mean(axis=0).mean(axis=0)


def dark_pixels(img):
    count = 0
    h, w, c = img.shape
    for i in range(h):
        for j in range(w):
            count += 1 if img[i][j].mean() <= 60 else 0
    return count / (h * w), count



def median_darkness(img):
    total_dark_count = 0
    h_, w_, c_ = img.shape
    dark_values = []
    for step_h in range(0, 300, 75):
        for step_w in range(0, 500, 125):
            cropped = img[step_h:step_h+75, step_w:step_w+125]
            res, count = dark_pixels(cropped)
            dark_values.append(res)
            total_dark_count += count
    dark_values = np.array(dark_values)
    
    return np.median(dark_values), total_dark_count / (h_ * w_)



def extract_features(img, print_report=False):
    mean_c = mean_color(img)
    median, dark = median_darkness(img)
    
    r, g, b = mean_c
    
    if print_report:
        print('Mean color: ', mean_c)
        print('Dark %:     ', dark)
        print('Median:     ', median)
    
    return r, g, b, dark, median


def build_classifier():
	features = pd.read_csv('features.csv')
	features = features.loc[:, ~features.columns.str.contains('^Unnamed')]
	best_params = pd.read_csv('best_params.csv')
	best_params = best_params.loc[:, ~best_params.columns.str.contains('^Unnamed')]
	clf = MultiOutputRegressor(DecisionTreeRegressor()).fit(features, best_params)
	return clf


def get_opencv_count(img, clf):
    img = planarize(img)
    orig = img.copy()
    shape = (500, 300)
    img = cv2.resize(img, shape, interpolation = cv2.INTER_AREA)
    img_features = extract_features(img)
    params = list(clf.predict([img_features]).reshape(6))

    res = getCountHoughCircles(orig, params)
    return res


def get_cnt_preds_ocv():
	outputs_ocv = []
	clf = build_classifier()
	for i in test_df['ImageId']:
	    path = os.path.join('train/', str(i)+ ".jpg")
	    img = cv2.imread(path)
	    img = planarize(img)
	    count = get_opencv_count(img, clf)
	    outputs_ocv.append(count)
	return outputs_ocv