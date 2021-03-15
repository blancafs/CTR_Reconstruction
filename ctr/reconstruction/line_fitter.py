import cv2
import os
import pandas as pd
import numpy as np
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt
import math
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import PolynomialFeatures

# given contour shapes from backsubtract class, define polynomials going neatly through points in order
# to better represent the robot and have a better shot at matching correspondance.

def fitPolyToContours(contour):
    contlines = []

    for c in contour:
        approx = cv2.approxPolyDP(c, 0.009 * cv2.arcLength(c, False), 3, True)
        contlines.append(approx)

    return contlines


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
        cleancont = np.array(c).squeeze()
        xs = [x[0] for x in cleancont]
        ys = [x[1] for x in cleancont]
        finalxs.append(xs)
        finalys.append(ys)

    # provides clean x and y corresponding coordinates
    finalxs = np.asarray([item for sublist in finalxs for item in sublist])
    finalys = np.asarray([item for sublist in finalys for item in sublist])

    return [finalxs, finalys]


def ransac(pts_x, pts_y, n_iter=10, dist_thresh=15):
    best_m = 0
    best_c = 0
    best_count = 0

    # set up figure and ax
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.scatter(pts_x, pts_y, c='blue')

    plt.ion()

    for i in range(n_iter):

        print("iteration: ", str(i))
        random_x1 = 0
        random_y1 = 0
        random_x2 = 0
        random_y2 = 0

        # select two unique points
        while (random_x1 == random_x2) or (random_y1 == random_y2):
            index1 = np.random.choice(pts_x.shape[0])
            index2 = np.random.choice(pts_x.shape[0])
            random_x1 = pts_x[index1]
            random_y1 = pts_y[index1]
            random_x2 = pts_x[index2]
            random_y2 = pts_y[index2]

        print("random point 1: ", random_x1, random_y1)
        print("random point 2: ", random_x2, random_y2)

        # slope and intercept for the 2 points
        if (random_x2 - random_x1 == 0) and (random_y2 - random_y1 != 0):
            continue
        m = (random_y2 - random_y1) / (random_x2 - random_x1)
        c = random_y1 - m * random_x1
        count = 0
        for i, value in enumerate(pts_x):

            # calculate perpendicular distance between sample line and input data points
            dist = abs(-m * pts_x[i] + pts_y[i] - c) / math.sqrt(m ** 2 + 1)

            # count the number of inliers
            if dist < dist_thresh:
                count = count + 1

        print("Number of inliers: ", count)

        # best line has the maximum number of inliers
        if count > best_count:
            best_count = count
            best_m = m
            best_c = c

        ax.scatter([random_x1, random_x2], [random_y1, random_y2], c='red')

        # draw line between points
        line = ax.plot([0, 1000], [c, m * 1000 + c], 'red')
        plt.draw()
        plt.pause(1)
        line.pop(0).remove()
        ax.scatter([random_x1, random_x2], [random_y1, random_y2], c='blue')

    print("best_line: y = {1:.2f} x + {1:.2f}".format(m, c))

    ax.plot([0, 1000], [best_c, best_m * 1000 + best_c], 'green')
    plt.ioff()
    plt.show()


def fitCurve(img, xs, ys):
    # def func(x, a, b, c):
    #     return a * np.square(x) + b * x + c

    # Choose best polynomial accordint to error in curve fit
    def findFunction(x, y, pmin=1, pmax=5):
        # a * np.exp(-b * x) + c

        def buildPoly(n):

            the_args = (n + 1) * []

            def poly(x, *args):
                if len(args) < n + 1:
                    print('Not enough args passed')
                    return 0
                res = 0
                for i in range(n + 1):
                    res += np.power(x, i) * args[-i]

            return poly

        errors = {}
        popts = {}
        for degree in range(pmin, pmax + 1):
            current_func = buildPoly(degree)
            popt, pcov = curve_fit(current_func, xs, ys)
            err = np.sqrt(np.diag(pcov))
            errors[degree] = err
            popts[degree] = popt
            print("Trying polynomial ", degree, ", error:", err, ", with parameters:", popt)
            print(err)

        # Error check and pick best N
        bestError = min(errors, key=errors.get)
        return popts[bestError], buildPoly(bestError)

    plt.imshow(img)
    plt.plot(xs, ys, 'b-', label='data')

    popts, func = findFunction(xs, ys)

    plt.plot(xs, func(xs, *popts), 'r-')

    plt.xlabel('x')
    plt.ylabel('y')
    plt.legend()
    plt.show()


def polyFitting(img_idx, cam, xs, ys, deg):
    xs = xs.reshape((len(xs), 1))
    ys = ys.reshape((len(ys), 1))

    df = pd.DataFrame(columns=['Cam', 'Image_idx', 'degree', 'RMSE', 'R2'])

    polynomial_features = PolynomialFeatures(degree=deg)
    x_poly = polynomial_features.fit_transform(xs)

    model = LinearRegression()
    model.fit(x_poly, ys)
    y_poly_pred = model.predict(x_poly)

    rmse = np.sqrt(mean_squared_error(ys, y_poly_pred))
    r2 = r2_score(ys, y_poly_pred)

    # df_entry = [cam, img_idx, deg, rmse, r2]
    # df.loc[len(df)] = df_entry
    #
    # # plt.show()
    # print("Saving to csv file for cam", str(cam), "...")
    # filename = "C:\\Users\\Blanca\\Documents\\CTRTracking\\threed_reconstruct\\poly_results.csv"
    # with open(filename, 'a') as f:
    #     df.to_csv(f, header=f.tell() == 0, index=False)

#    return the coordinates for the polynomial fit

    # print('xpoly shape', xs.shape)
    # print('y_poly_pred shape', y_poly_pred.shape)
    return xs, y_poly_pred


def create_polynomial_regression_model(degree, xs, ys):
    "Creates a polynomial regression model for the given degree"

    xs = xs.reshape((len(xs), 1))
    ys = ys.reshape((len(ys), 1))

    poly_features = PolynomialFeatures(degree=degree)

    # transforms the existing features to higher degree features.
    X_train_poly = poly_features.fit_transform(xs)

    # fit the transformed features to Linear Regression
    poly_model = LinearRegression()
    poly_model.fit(X_train_poly, ys)

    # predicting on training data-set
    y_train_predicted = poly_model.predict(X_train_poly)

    # predicting on test data-set
    y_test_predict = poly_model.predict(poly_features.fit_transform(xs))

    # evaluating the model on training dataset
    rmse_train = np.sqrt(mean_squared_error(ys, y_train_predicted))
    r2_train = r2_score(ys, y_train_predicted)

    # evaluating the model on test dataset
    rmse_test = np.sqrt(mean_squared_error(ys, y_test_predict))
    r2_test = r2_score(ys, y_test_predict)

    print("The model performance for the training set")
    print("-------------------------------------------")
    print("RMSE of training set is {}".format(rmse_train))
    print("R2 score of training set is {}".format(r2_train))

    print("\n")

    print("The model performance for the test set")
    print("-------------------------------------------")
    print("RMSE of test set is {}".format(rmse_test))
    print("R2 score of test set is {}".format(r2_test))


class LineFitter:

    def __init__(self, method='polyreg'):
        self.method = method

    def fitLine(self, image, i, cam, conts):

        if self.method == 'fitpoly':
            contlines = fitPolyToContours(conts)
            showFittedLine(image, i, contlines)

        if self.method == 'scipy':
            [xs, ys] = concatContCoors(conts)
            fitCurve(image, xs, ys)

        if self.method == 'polyreg':
            [xs, ys] = concatContCoors(conts)
            xs, ys = polyFitting(i, cam, xs, ys, 5)
            return xs, ys
            # create_polynomial_regression_model(4, xs, ys)

    def fitLines(self, images, cam, contours: dict, save_folder=""):
        message = False
        for i, cont in contours.items():
            if i > 3:

                # start at index 4 as before alg does not have time to be accurate
                xs, ys = self.fitLine(images[i], i, cam, cont)

                xs = xs.tolist()
                ys = ys.tolist()

                df = pd.DataFrame({'X': xs, 'Y': ys}, index=list(range(len(xs))))

                filename = 'poly_points_cam'+ str(cam) + '_' + str(i) +".csv"

                if len(save_folder) > 0:
                    save_path = os.path.join(save_folder, filename)
                    with open(save_path, 'a') as f:
                        df.to_csv(f, header=f.tell() == 0, index=False)
                    message = True
        return message

# ===================================================================================================================================================================
