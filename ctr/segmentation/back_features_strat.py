from ctr.segmentation.segmentor import SegmentationStrategyAbstract
from ctr.segmentation.utils import *


import matplotlib.pyplot as plt
from scipy import ndimage
import pandas as pd
import time


class FinalStrategy(SegmentationStrategyAbstract):
    def segmentImg(self, img):
        self.info('Frame differentiation segmentation for img called.')

    def segmentImgs(self, arr):
        times = []
        """
            1. apply cnt subtractor
            2. dilate and apply mask to find SIFT points
        """

        background = cv2.imread("C:\\Users\\Blanca\\Desktop\\CTRTracking\\data\\background\\cam1_back1.png")
        cnt_sub = cv2.bgsegm.createBackgroundSubtractorCNT(minPixelStability=2)

        oldkeypoints = []
        olddescriptors = []

        # these methods take care of frame differencing, gaussian filtering and thresholding

        i = 0
        cnt_sub.apply(background)
        finalcoors = []

        # dataframe to keep feature points
        df = pd.DataFrame(columns=['image_number', 'feature_coors'])

        # loop includes every frame in list in the subtractor, works well after 2/3 frames
        while i < len(arr):

            times.append(time.perf_counter())

            # print('Managing image number ', i, 'out of ', len(arr)-1)
            frame = arr[i]
            mask = cnt_sub.apply(frame)

            # first mask always empty so apply and skip
            if i == 0:
                i += 1
                continue

            robot = cv2.bitwise_and(frame, frame, mask=mask)
            gray = cv2.cvtColor(robot, cv2.COLOR_BGR2GRAY)
            contours, _ = cv2.findContours(gray, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
            contours = [c.astype(int) for c in contours if cv2.contourArea(c) > 80]
            # robot_conts = cv2.drawContours(robot, contours, -1, (0, 255, 0), 2)

            if len(contours) > 4:
                # print("skipped ", i)
                # print('Too many contours, assume more than one robot in frame...')
                i += 1
                continue

            else:  # if only one robot in image
                # print('Lonely robot found, finding coors and dilating...')

                # use empty image to put down contours and dilate for mask
                empty = np.zeros(gray.shape, np.uint8)

                contimg = cv2.drawContours(empty, contours, -1, 255, -1)
                struct2 = ndimage.generate_binary_structure(2, 2)
                dilated = ndimage.binary_dilation(contimg, structure=struct2, iterations=3).astype(contimg.dtype)

                masked = cv2.bitwise_and(frame, frame, mask=dilated)
                masked = cv2.cvtColor(masked, cv2.COLOR_BGR2GRAY)

                # returned in the format: extLeft, extRight, extTop, extBot
                extrema = get_extrema_coors(contours)
                finalextrema = get_overall_extrema(extrema)
                finalextrema = filter_coordinates(finalextrema)
                finalextrema = join_valid_coordinates(finalextrema)

                probs = get_last_probabilities(contimg, finalextrema)
                maxx = -1
                maxi = -1
                for k, p in enumerate(probs):
                    if p > maxx:
                        maxi = k
                        maxx = p

                end = finalextrema[maxi]


                # keep top and bottom coordinates for all - how to know it's the end?

                # plt.imshow(masked)
                # plt.scatter(end[0], end[1])
                # plt.show()

                # Apply feature finding to image and next image, find matching SIFT features
                orb = cv2.ORB_create()
                newkeypoints, newdescriptors = orb.detectAndCompute(masked, None)
                newkeypoints, newdescriptors = filter_keypoints(newkeypoints, newdescriptors)

                cleanpoints = [tuple(point.pt) for point in newkeypoints]

                newrow = {'image_number': i, 'feature_coors': cleanpoints}

                df = df.append(newrow, ignore_index=True)

                if len(olddescriptors) < 1 or len(oldkeypoints) < 1:
                    oldkeypoints = newkeypoints
                    olddescriptors = newdescriptors
                    i += 1
                    continue

                # Creating BFMatcher object with distance measure specified, crosscheck parameter increases accuracy
                bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=True)
                matches = bf.match(np.asarray(olddescriptors), np.asarray(newdescriptors))

                # Sort them in order of distance
                matches = sorted(matches, key=lambda x: x.distance)

                src_pts = np.float32([oldkeypoints[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
                dst_pts = np.float32([newkeypoints[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)

                # Draw first 10 matches.
                matched = cv2.drawMatches(arr[i-1], oldkeypoints, frame, newkeypoints, matches[:10], None, flags=2)

                print('Showing matched results...')
                plt.title('Matching ORB Features')
                plt.imshow(matched), plt.show()

                oldkeypoints = newkeypoints
                olddescriptors = newdescriptors

            i += 1

        # print("saving df")
        # df.to_csv('cam2_keypoints.csv', index=False)

        # df = pd.DataFrame()
        # df['times'] = times
        #
        # files = os.listdir('time_measurements_final')
        # maxx = len(files)
        #
        # df.to_csv('time_measurements_final/final' + str(maxx) + '.csv', index=False)
        #
        # # print(finalcoors)
        # print('finito')
