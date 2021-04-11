from ctr.reader.reader import Reader
from ctr.segmentation import BackgroundSubstraction
from ctr.reconstruction import *
import numpy as np
import time


if __name__ == '__main__':
    reader = Reader()

    # Read original camera images and background
    cam1imgs, cam2imgs, backimg = reader.read_images(CAM1_FOLDER, CAM2_FOLDER, BACKG_FOLDER)

    # Apply segmentation from part 1 to each set
    segmentor = BackgroundSubstraction()
    contours_cam1 = segmentor.segmentImgs(cam1imgs, backimg)
    contours_cam2 = segmentor.segmentImgs(cam2imgs, backimg)

    # Fit polynomials to robot body in order to minimise points to transform, save to csvs
    lf = LineFitter()
    cam1_lf = lf.fitLines(cam1imgs, 1, contours_cam1, save_folder=LF_RESULTS_FOLDER)
    cam2_lf = lf.fitLines(cam2imgs, 2, contours_cam2, save_folder=LF_RESULTS_FOLDER)

    good_fits = np.arange(4, 18)

    good_coor_sets = {}
    for k in good_fits:
        corresp = [cam1_lf[k], cam2_lf[k]]
        good_coor_sets.update({k : corresp})

    wg = WeightedGraph()
    # graph_matches = GRAPH_MATCHES
    graph_matches = {}
    print('Applying weighted graph...')
    # Apply weighted graph matching to sets of coordinates
    for idx in good_coor_sets.keys():
        c1 = good_coor_sets.get(idx)[0]
        c2 = good_coor_sets.get(idx)[1]
        wg.set_pairs(c1, c2)
        row_idx, col_idx = wg.solve()
        matching = [row_idx, col_idx]
        graph_matches.update({idx: matching})
        wg.clear()
        print((idx-4) / len(good_coor_sets.keys()))

    # SAVING GRAPH MATCHES (for truth data)
    # a_file = open("test/truth_matches_contours.pkl", "wb")
    # pickle.dump(graph_matches, a_file)
    # a_file.close()

    # returns dict of idx : [c1coors, c2coors]
    corresp_coors = join_corresp_coors(good_coor_sets, graph_matches)

    # TRANSFORM CORRESPONDING COORDINATES
    robots = transform_many(corresp_coors)

    print('showing robot...')
    show_robot_3d(robots)

    # SHOW REFLECTION ERROR
    for c in corresp_coors.keys():
        coors = corresp_coors.get(c)
        c2lf = cam2_lf[c]
        cam1s = [x[0] for x in coors]
        cam2s = [x[1] for x in coors]

        err_sum, c2prj = error_function(BEST_TRANSF_X, cam1s, cam2s)
        print('Image ', str(c), ' had error:', str(err_sum))
        show_reflection(cam2imgs[c], c2prj, c)

    # SAVE 3D COORDINATES OF ROBOT
    # for key in robots.keys():
    #     name = f'final_rob{key}.mat'
    #     np.savetxt(name, robots[key])

    # print('Segmentation time: ', segmentation_time)
    # print('Curve fitting time: ', curve_fit_time)
    # print('Graph time: ', graph_time)
    # print('Transform time: ', transform_time)



