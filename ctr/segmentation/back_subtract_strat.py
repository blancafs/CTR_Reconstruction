from ctr.segmentation.segmentor import SegmentationStrategyAbstract
from ctr.segmentation.utils import *

# Outside imports
import cv2
import time


class BackgroundSubstraction(SegmentationStrategyAbstract):
    def segmentImg(self, img):
        # print('Frame differentiation segment img called.')
        pass

    def segmentImgs(self, arr, back_img):
        # print('Frame differentiation segment imgs called...')
        times = []
        """
            1. apply cnt subtractor
            2. if more than one robot shape, differ from last frame to remove one body
            CNT is faster than MOG2 and resilient to light changing! https://github.com/sagi-z/BackgroundSubtractorCNT
        """
        background = back_img
        # these methods take care of frame differencing, gaussian filtering and thresholding
        cnt_sub = cv2.bgsegm.createBackgroundSubtractorCNT(minPixelStability=2)

        i = 0

        cnt_sub.apply(background)
        # holds the tip measurements for each image given
        coors = {}

        # whole coors
        body_coors = {}

        times.append(time.perf_counter())

        # loops through all frames passed by main
        while i < len(arr):
            # print('Image..', str(i))
            times.append(time.perf_counter())

            frame = arr[i]
            mask = cnt_sub.apply(frame)
            robot = cv2.bitwise_and(frame, frame, mask=mask)
            gray = cv2.cvtColor(robot, cv2.COLOR_BGR2GRAY)
            contours, _ = cv2.findContours(gray, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            contours = [c.astype(int) for c in contours if cv2.contourArea(c) > 80]
            robot_conts = cv2.drawContours(robot, contours, -1, (0, 255, 0), 3)
            #
            # cv2.imshow('Contours', robot_conts)
            # cv2.waitKey(0)
            # cv2.destroyAllWindows()

            if len(contours) > 4:
                # print("skipped ", i)
                # print('Too many contours, assume more than one robot in frame...')
                i += 1
                continue

            else:

                # save all body coordinates to body_coors for 3d reconstruction
                body_coors.update({i: contours})

                # # CODE TO GET JUST THE END TIP ###################################
                # # returned in the format: extLeft, extRight, extTop, extBot
                # extrema = get_extrema_coors(contours)
                # finalextrema = get_overall_extrema(extrema)
                # finalextrema = filter_coordinates(finalextrema)
                # finalextrema = join_valid_coordinates(finalextrema)
                #
                # contimg = cv2.cvtColor(robot_conts, cv2.COLOR_BGR2GRAY)
                # # plt.imshow(contimg), plt.show()
                #
                # probs = get_last_probabilities(contimg, finalextrema)
                # maxx = -1
                # maxi = -1
                # for k, p in enumerate(probs):
                #     if p > maxx:
                #         maxi = k
                #         maxx = p
                #
                # end = finalextrema[maxi]
                #
                # # plt.title(str(i)), plt.scatter(end[0], end[1]), plt.imshow(frame), plt.show()
                #
                # coors.update({i: end})
                ######################################################################

            i += 1

        # for whole body contours
        return body_coors

        # CODE FOR SAVING TIP COOR TO CSV ############################
        # df = pd.DataFrame()
        # df['image'] = coors.keys()
        # df['coordinates'] = [tuple(b) for (a, b) in coors.items()]
        #
        # files = os.listdir('back_tipmeasurements')
        # maxx = len(files)
        #
        # cam = 'cam1'
        # # if maxx % 2 > 0:
        # #     cam = 'cam2'
        #
        # df['camera'] = cam
        #
        # print(df)
        # df.to_csv('back_tipmeasurements/' + cam + '_' + str(maxx) + '.csv', index=False)

        # df = pd.DataFrame()
        # df['times'] = times
        #
        # files = os.listdir('time_measurements_background')
        # maxx = len(files)
        #
        # df.to_csv('time_measurements_background/back' + str(maxx) + '.csv', index=False)
        #
        # print('finito')

        # # for end tip measurements
        # return coors


