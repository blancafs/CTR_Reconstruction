from ctr.segmentation.segmentor import SegmentationStrategyAbstract
from ctr.segmentation.utils import *

# outside imports
import cv2
import numpy as np
import time
import os
import pandas as pd
import matplotlib.pyplot as plt


class EdgeDetection(SegmentationStrategyAbstract):

    def segmentImg(self, img):
        # print('EdgeDetection dealing with image...')

        '''
        To pre-process the image, it is:
            1. Blurred using a gaussian filter with a 5,5 kernel
            2. Auto canny function for best canny settings applied
            3. Image is eroded and dilated to extract rough shape of robot
            4. Dilated to allow space for features later (?)
            5. Find contours of image
        '''

        # Using canny for blurring and edge detection
        blur = cv2.GaussianBlur(img, (5, 5), 0)
        gray = cv2.cvtColor(blur, cv2.COLOR_BGR2GRAY)
        canny = auto_canny(gray)
        opening = cv2.morphologyEx(canny, cv2.MORPH_OPEN, (1, 1))
        kernel = np.ones((7, 7), np.float32) / 25
        mask = cv2.filter2D(opening, -1, kernel)
        contours, _ = cv2.findContours(mask.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        '''
        Once robot body segmented:
            1. Get extrema points of biggest contour and closest contour (if robot fragmented due to reflection from lamp.)
            2. Join contours using extrema points
            3. Join all contours and fill in
            4. Apply mask to original image
        '''
        fractured = False  # assume robot full unless other valid contours found

        # Merge contours - big n closest returns the first biggest contour and the closest contour to that
        big_nclosest = big_closest_contour(contours)

        # empty mask to fill in
        empty = np.zeros(gray.shape, np.uint8)

        if len(big_nclosest) > 1:
            fractured = True

        if fractured:
            # get coordinates for bottom, left, right points of each contour
            extrema_coors = get_extrema_coors(big_nclosest)

            # Joins bottom of small contour and bottom of big contour
            image = cv2.line(gray, extrema_coors[0][3], extrema_coors[1][3], (0, 255, 0), 10)
            image = cv2.drawContours(image, big_nclosest, -1, (0, 255, 0), 3)
            newcontours, _ = cv2.findContours(image, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

            # Joining contours and selecting merged, eliminating too big contours
            newcontours = [c for c in newcontours if cv2.contourArea(c) < 10000]
            areas = [cv2.contourArea(c) for c in newcontours]
            fin_idx = np.argmax(areas)  # final contour index holding whole robot

            cont = [newcontours[fin_idx]]
            mask = cv2.drawContours(empty, cont, -1, (255, 255, 255), -1)

        else:  # if robot is whole, apply biggest contour as mask
            cont = [big_nclosest[0]]
            mask = cv2.drawContours(empty, [big_nclosest[0]], -1, (255, 255, 255), -1)

        extrema = get_extrema_coors(cont)
        finalextrema = filter_coordinates(extrema[0])
        finalextrema = join_valid_coordinates(finalextrema)

        probs = get_last_probabilities(mask, finalextrema)
        if len(probs)<1:
            return -1
        maxx = -1
        maxi = -1
        for k, p in enumerate(probs):
            if p > maxx:
                maxi = k
                maxx = p

        end = finalextrema[maxi]

        plt.scatter(end[0], end[1]), plt.imshow(img), plt.show()

        '''
        Returns coordinates of end point.
        '''
        return end

        # print(len(keypoints))
        # img_points = cv2.drawKeypoints(grayimg,keypoints,None,(255,0,0),0)
        # plt.imshow(img_points),plt.show()

    def segmentImgs(self, list):
        # print('EdgeDetection dealing with images...')
        # times = []
        coors = {}

        for i in range(len(list)):
            # times.append(time.perf_counter())
            img = list[i]
            print(i)
            end = self.segmentImg(img)
            if end == -1:
                print('fail)')
            else:
                coors.update({i: end})

        df = pd.DataFrame()
        df['image'] = coors.keys()
        df['coordinates'] = [tuple(b) for (a, b) in coors.items()]

        files = os.listdir('edge_tipmeasurements')
        maxx = len(files)

        cam = 'cam1'
        # if maxx % 2 > 0:
        #     cam = 'cam2'

        df['camera'] = cam

        print(df)
        df.to_csv('edge_tipmeasurements/' + cam + '_' + str(maxx) + '.csv', index=False)
        return coors


        # df = pd.DataFrame()
        # df['times'] = times
        #
        # files = os.listdir('time_measurements_edge')
        # maxx = len(files)
        #
        # df.to_csv('time_measurements_edge/edge' + str(maxx) + '.csv', index=False)
