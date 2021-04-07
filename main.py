from ctr.reader.reader import Reader
from ctr.segmentation import BackgroundSubstraction
from ctr.reconstruction import *
import numpy as np


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

    print(len(cam1_lf[5]), len(cam2_lf[5]))

    good_fits = [5]
        # np.arange(4, 19)
    good_coor_sets = {}

    for gf in good_fits:
        corresp = [cam1_lf[gf],  cam2_lf[gf]]
        good_coor_sets.update({gf : corresp})

    # good_coor_sets = {}
    # for k in cam1_lf.keys():
    #     corresp = [cam1_lf[k], cam2_lf[k]]
    #     good_coor_sets.update({k : corresp})

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
        print((idx - 4) / len(good_coor_sets.keys()))

    # print(graph_matches.items())

    # rerturns dict of idx : [c1coors, c2coors]
    corresp_coors = join_corresp_coors(good_coor_sets, graph_matches)

    # error = error_function(BEST_TRANSF_X, p1s, p2s, mode='mult')
    # errs, cam2reflect = get_c2reflection(corresp_coors)
    # show_reflection(cam2imgs, cam2reflect)

    # robots = transform_many(corresp_coors)
    # show_robot_3d(robots)
    for c in corresp_coors.keys():
        print('in loop...')
        coors = corresp_coors.get(c)
        c2lf = cam2_lf[c]
        cam1s = [x[0] for x in coors]
        cam2s = [x[1] for x in coors]

        err_sum, c2prj = error_function(BEST_TRANSF_X, cam1s, cam2s)
        print(c2prj)
        show_reflection(cam2imgs[c], c2prj, c)
    # #
    # # for key in robots.keys():
    # #     name = f'lfscipy3_rob{key}.mat'
    # #     np.savetxt(name, robots[key])
    #
    #
    #
