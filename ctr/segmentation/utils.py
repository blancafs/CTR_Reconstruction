import numpy as np
import cv2
from statistics import mean
import math
from scipy.special import softmax

def auto_canny(image, sigma=0.33):
    # compute the median of the single channel pixel intensities
    v = np.median(image)
    # apply automatic Canny edge detection using the computed median
    lower = int(max(0, (1.0 - sigma) * v))
    upper = int(min(255, (1.0 + sigma) * v))
    edged = cv2.Canny(image, lower, upper)
    # return the edged image
    return edged


# Will take length of robot and current contours found, if length
# of main contour smaller than expected length, find closest contour
def big_closest_contour(contours, length=-1):
    if len(contours) < 0:
        return -1
    maxarea = -1  # will hold area of main, biggest, contour
    maxcontour_idx = -1
    centres = []  # will hold centres of all contours

    # Ignoring too small contours
    contours = [c for c in contours if cv2.contourArea(c) > 200]
    if len(contours) < 1:
        # print('Error, no contours found!')
        pass

    # Get centres of all contours and select biggest
    for i, c in enumerate(contours):
        M = cv2.moments(c)
        cx = int(M['m10'] / M['m00'])
        cy = int(M['m01'] / M['m00'])
        centres.append([cx, cy])
        if cv2.contourArea(c) > maxarea:
            maxarea = cv2.contourArea(c)
            maxcontour_idx = i
            maxcontour = c

    # distances from valid contours to main contour
    distances = [np.linalg.norm(np.array(a) - np.array(centres[maxcontour_idx])) for a in centres if
                 a != centres[maxcontour_idx]]

    # remove biggest contour
    del contours[maxcontour_idx]

    # if distances not empty, aka there are valid small contours apart from the big one, return max and closest
    if len(distances) > 0:
        closest = contours[distances.index(min(distances))]  # select contour closest to main
        return [maxcontour, closest]

    # if distances empty it means no valid small contour, meaning robot shape must be full. returns max contour only
    else:
        return [maxcontour]


def midpoint(p1, p2):
    return [(p1[0] + p2[0]) / 2, (p1[1] + p2[1]) / 2]


def join_contours(cont1, cont2, segments=8):
    """
    Given 2 points (j1, and j2) calculate the bezier curve
    between them - using point pa1 to pull  the curve away
    from the points.

    Segments determines the number of points calculated for
    the curve.

    Format of data given: left, right, top, bottom
    """
    # Holds most left point of both contours given
    left1 = cont1[3]
    left2 = cont2[3]

    leftmid = midpoint(left1, left2)
    # rightmid = midpoint(right1, right2)

    step = 1.0 / segments

    # MAKING CURVE JOINING LEFT POINTS
    leftcurve = []
    leftcurve.append(left1)
    # Will create quadratic bezier curve between left and right points
    for t in np.arange(step, 1.0 - step, step):
        x = ((1 - t) * (1 - t) * left1[0] + 2 * (1 - t) * t * leftmid[0] + t * t * left2[0])
        y = ((1 - t) * (1 - t) * left1[1] + 2 * (1 - t) * t * leftmid[1] + t * t * left2[1])
        leftcurve.append((x, y))
    leftcurve.append(left2)
    return leftcurve


def get_extrema_coors(contours):
    extrema_coors = []
    for c in contours:
        if len(c) > 0:
            extLeft = tuple(c[c[:, :, 0].argmin()][0])
            extRight = tuple(c[c[:, :, 0].argmax()][0])
            extTop = tuple(c[c[:, :, 1].argmin()][0])
            extBot = tuple(c[c[:, :, 1].argmax()][0])
            extrema_coors.append([extLeft, extRight, extTop, extBot])
    return extrema_coors


def get_overall_extrema(extremapoints):
    coordinates = []
    for x in extremapoints:
        for y in x:
            coordinates.append(y)

    extLeft = (1000, 1000)
    extRight = (-1, -1)
    extTop = (1000, 1000)
    extBot = (-1, -1)
    for c in coordinates:
        if c[0] < extLeft[0]:
            extLeft = c
        elif c[0] > extRight[0]:
            extRight = c
        elif c[1] < extTop[1]:
            extTop = c
        elif c[1] > extBot[1]:
            extBot = c
    return [extLeft, extRight, extTop, extBot]


def filter_keypoints(keypoints, descriptors):
    # Removes points in opening in the centre of img
    keyps = []
    descrips = []
    xrange = list(range(312, 335))
    yrange = list(range(128, 151))

    removeidxs = []
    for i, k in enumerate(keypoints):
        x = k.pt[0]
        y = k.pt[1]
        if 311 < x < 339 and 128 < y < 160:
            removeidxs.append(i)
        # if 580 < x < 630 and 235 < y < 295:
        #     removeidxs.append(i)

    for i, x in enumerate(descriptors):
        if i not in removeidxs:
            keyps.append(keypoints[i])
            descrips.append(x.astype(np.float32))

    # print('Removed ', len(keypoints) - len(keyps), len(descriptors) - len(descrips))

    return keyps, descrips


def get_biggest_contour(contours):
    if len(contours) < 1:
        return -1

    maxidx = 0
    maxarea = -1
    for i, c in enumerate(contours):
        if cv2.contourArea(c) > maxarea:
            maxarea = cv2.contourArea(c)
            maxidx = i

    maxcontour = contours[maxidx]
    return maxcontour


def get_last_probabilities(img, extrema):
    # plt.imshow(img), plt.show()
    tendencies = []
    extrema = filter_coordinates(extrema)
    for e in extrema:
        neighbourhood, new_coor = calculate_neighbourhood(img, e)
        ten = get_tendency(neighbourhood, new_coor)
        tendencies.append(ten)
    if len(tendencies)>0:
        probs = list(softmax(tendencies))
    else:
        probs = []
    return probs


def get_tendency(neighbourhood, new_coor):
    # calculate one sidedness of neigbourhood and point
    x = new_coor[0]
    y = new_coor[1]

    vectors = []
    mag = -1
    for curry, row in enumerate(neighbourhood):
        for currx, pix in enumerate(row):
            if pix != 0:
                vectors.append([currx - x, curry - y])
    if len(vectors) > 0:
        avgx = mean([x[0] for x in vectors])
        avgy = mean([x[1] for x in vectors])
        mag = math.sqrt(avgx * avgx + avgy * avgy)

    return mag


def calculate_neighbourhood(img, point, n=20):
    x_offset = n
    y_offset = n

    limx = img.shape[0]
    limy = img.shape[1]
    # given a point return the matrix that is the neighbourhood
    minx = point[0] - n
    maxx = point[0] + n
    miny = point[1] - n
    maxy = point[1] + n
    if minx < 0:
        x_offset = x_offset + minx
        minx = 0
    elif maxx >= limx:
        maxx = limx - 1
    if miny < 0:
        y_offset = y_offset + miny
        miny = 0
    elif maxy >= limy:
        maxy = limy - 1

    neigh = img[int(miny):int(maxy), int(minx):int(maxx)]
    return neigh, (x_offset, y_offset)


def filter_coordinates(extrema):
    finalextrema = []
    for e in extrema:
        x = e[0]
        y = e[1]
        if (307 < x < 345 and 120 < y < 170) or x == 0 or y == 0:
            continue
        # if 580 < x < 630 and 235 < y < 295 or x == 0 or y == 0:
        #     continue
        else:
            finalextrema.append(e)

    return finalextrema


# Takes set of coordinates and if some are close enough they are joined. Purpose: join end coordinates.
def join_valid_coordinates(extrema, distance=5):
    if len(extrema)<3:
        # print('Given empty extrema list ...')
        return extrema
    finalextrema = []
    dist = {}
    merged = []

    first = 0
    second = first + 1

    while True:
        if second == len(extrema):
            first += 1
            second = first + 1
        if first == len(extrema) - 1:
            break
        xdis = abs(extrema[first][0] - extrema[second][0])
        ydis = abs(extrema[first][1] - extrema[second][1])
        if xdis <= distance and ydis <= distance:
            dist.update({first: second})
        second += 1

    for k in dist.keys():
        x = (extrema[k][0] + extrema[dist.get(k)][0]) / 2
        y = (extrema[k][1] + extrema[dist.get(k)][1]) / 2
        finalextrema.append((x, y))
        merged.append(k)
        merged.append(dist.get(k))

    rest = [x for i, x in enumerate(extrema) if i not in merged]

    finalextrema = finalextrema + rest
    return finalextrema
