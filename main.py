from ctr.reader.reader import Reader
from ctr.resources import *
from ctr.segmentation import BackgroundSubstraction
from ctr.reconstruction import LineFitter


if __name__ == '__main__':
    cam1imgs, cam2imgs, backimg = Reader().read(CAM1_FOLDER, CAM2_FOLDER, BACKG_FOLDER)

    segmentor = BackgroundSubstraction()
    contours_cam1 = segmentor.segmentImgs(cam1imgs, backimg)
    contours_cam2 = segmentor.segmentImgs(cam2imgs, backimg)

    lf = LineFitter()
    success_cam1 = lf.fitLines(cam1imgs, 1, contours_cam1, save_folder=LF_RESULTS_FOLDER)
    success_cam2 = lf.fitLines(cam2imgs, 2, contours_cam2, save_folder=LF_RESULTS_FOLDER)


