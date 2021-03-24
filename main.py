import plotly

from ctr.reader.reader import Reader
from ctr.resources import *
from ctr.segmentation import BackgroundSubstraction
from ctr.reconstruction import LineFitter, WeightedGraph, transform, transform_many, show_robot_3d, join_corresp_coors
import numpy as np
from scipy.io import savemat

import matplotlib.pyplot as plt
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.express as px


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

    good_fits = [5, 6, 7, 8]
    good_coor_sets = {}

    for gf in good_fits:
        corresp = [cam1_lf[gf],  cam2_lf[gf]]
        good_coor_sets.update({gf : corresp})

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

    robots = transform_many(corresp_coors)
    show_robot_3d(robots)



