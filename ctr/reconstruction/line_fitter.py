import collections

import cv2
import numpy as np
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, r2_score
from scipy.optimize import curve_fit
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import (
    LinearRegression, TheilSenRegressor, RANSACRegressor, HuberRegressor)
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline


# given contour shapes from backsubtract class, define polynomials going neatly through points in order
# to better represent the robot and have a better shot at matching correspondance.

def showFittedLine(image, i, approxpoly):
    # draws boundary of contours to check resultssss
    # for line in approxpoly:
    for line in approxpoly:
        cv2.drawContours(image, [line], 0, (0, 0, 255), 5)

    # fname = './line-fittingresults/' + str(i) + '.png'
    # cv2.imwrite(fname, image)
    cv2.imshow('Contours', image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def concatContCoors(contours):
    finalxs = []
    finalys = []

    for c in contours:
        cleancont = c.squeeze()
        xs = [x[0] for x in cleancont]
        ys = [x[1] for x in cleancont]
        finalxs.append(xs)
        finalys.append(ys)

    # provides clean x and y corresponding coordinates
    finalxs = np.asarray([item for sublist in finalxs for item in sublist])
    finalys = np.asarray([item for sublist in finalys for item in sublist])

    return [finalxs, finalys]


def line_range(start, stop, count):
    step = (stop - start) / float(count)
    return np.array([start + i * step for i in range(count)])


def fitCurve(img, xs, ys, i, cam, plot=False):
    def objective(x, a, b, c, d):
        return a * x + b * x ** 2 + c * x ** 3 + d

    popt, _ = curve_fit(objective, xs, ys)
    a, b, c, d = popt

    x_line = line_range(min(xs), max(xs), 500)
    # , 500
    y_line = objective(x_line, a, b, c, d)

    # if plot:
    #     plt.scatter(xs, ys)
    #     plt.plot(x_line, y_line, '--', color='red')
    #     plt.imshow(img)
    #     plt.title(str(cam) + '_' + str(i))
    #     name = str(cam) + str(i) + '.png'
    #     plt.savefig(name)
    #     plt.show()

    return x_line, y_line


def huber_fit(i, img, xs, ys):
    "Creates a polynomial regression model for the given degree"
    x_scaler, y_scaler = StandardScaler(), StandardScaler()
    x_train = x_scaler.fit_transform(xs[..., None])
    y_train = y_scaler.fit_transform(ys[..., None])

    model = HuberRegressor(epsilon=1)
    model.fit(x_train, y_train.ravel())
    test_x = line_range(min(xs), max(xs), 500)

    predictions = y_scaler.inverse_transform(
        model.predict(x_scaler.transform(test_x[..., None]))
    )

    plt.scatter(xs, ys)
    plt.plot(test_x, predictions, '--', color='red')
    plt.imshow(img)
    plt.title(str(i))
    # name = str(cam) + str(i) + '.png'
    # plt.savefig(name)
    plt.show()

    return test_x, predictions


def clean_conts(xs, ys):
    coors = list(zip(xs, ys))
    coors = [list(a) for a in coors]
    nodobs = list(set(coors))
    sort = sorted(nodobs, key=lambda x: x[0])
    xs = [x[0] for x in sort]
    ys = [x[1] for x in sort]

    return xs, ys


def ransac(i, cam, img, xs, ys):
    np.random.seed(42)
    xs = xs[:, np.newaxis]

    x_test = line_range(min(xs), max(xs), 100)

    methods1 = {4: 'o', 5: 'o', 6: 'o', 7: 'o', 8: 'o', 9: 'o', 10: 'o', 11: 'r', 12: 'o', 13: 'o', 14: 'o', 15: 'o',
                16: 'o', 17: 'o', 18: 'o'}
    methods2 = {4: 'o', 5: 'o', 6: 'o', 7: 'o', 8: 'o', 9: 'o', 10: 'o', 11: 'r', 12: 'o', 13: 'o', 14: 'o', 15: 'r',
                16: 'o', 17: 'r', 18: 'o'}

    if cam == 1:
        met = methods1.get(i)
    if cam == 2:
        met = methods2.get(i)

    if met == 'o':
        estimator = LinearRegression()
    else:
        estimator = RANSACRegressor()

    model = make_pipeline(PolynomialFeatures(3), estimator)
    model.fit(xs, ys)
    y_plot = model.predict(x_test)
    # plt.plot(x_test, y_plot)
    # plt.imshow(img)
    # name = 'cam'+ str(cam) + '_' + str(i) + '.png'
    # plt.savefig(name)
    #
    #
    # plt.show()

    return x_test, y_plot


class LineFitter:

    def __init__(self, method='ransac', plot=True):
        self.method = method
        self.plot = plot

    def fitLine(self, image, i, cam, cont):

        if self.method == 'ransac':
            [xs, ys] = concatContCoors(cont)
            xs, ys = ransac(i, cam, image, xs, ys)
            return xs, ys

        if self.method == 'scipy':
            [xs, ys] = concatContCoors(cont)
            xs, ys = fitCurve(image, xs, ys, i, cam, self.plot)
            return xs, ys

        if self.method == 'huberfit':
            [xs, ys] = concatContCoors(cont)
            huber_fit(i, image, xs, ys)
            return xs, ys

        if self.method == 'none':
            [xs, ys] = concatContCoors(cont)
            [xs, ys] = clean_conts(xs, ys)
            print('xs', xs, 'ys', ys)
            return xs, ys

    def fitLines(self, images: list, cam: int, contours: dict, save_folder=""):

        fitted_lines = {}
        # idx : x,y...
        for i, cont in contours.items():
            if i > 3:
                # start at index 4 as before alg does not have time to be accurate
                xs, ys = self.fitLine(images[i], i, cam, cont)
                xs = np.array(xs).flatten()
                ys = np.array(ys).flatten()
                zipped = list(map(list, zip(xs, ys)))
                fitted_lines.update({i: zipped})

        return fitted_lines
