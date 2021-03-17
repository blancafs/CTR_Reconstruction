import plotly

from ctr.reader.reader import Reader
from ctr.resources import *
from ctr.segmentation import BackgroundSubstraction
from ctr.reconstruction import LineFitter, WeightedGraph, transform

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
    # lf = LineFitter()
    # success_cam1 = lf.fitLines(cam1imgs, 1, contours_cam1, save_folder=LF_RESULTS_FOLDER)
    # success_cam2 = lf.fitLines(cam2imgs, 2, contours_cam2, save_folder=LF_RESULTS_FOLDER)

    # Read csvs to retrieve poly points per image, pair by IMG_IDX = (CAM1COORS, CAM2COORS)
    dirname = LF_RESULTS_FOLDER
    poly_point_files = os.listdir(LF_RESULTS_FOLDER)
    cam1_files = [filename for filename in poly_point_files if 'cam1' in filename]
    cam2_files = [filename for filename in poly_point_files if 'cam2' in filename]

    image_stereo_coors = {}

    for i, c1_name in enumerate(cam1_files):
        for j, c2_name in enumerate(cam2_files):
            if i == j:
                coors1 = reader.read_poly_coors(dirname, c1_name)
                coors2 = reader.read_poly_coors(dirname, c2_name)
                image_coors = [coors1, coors2]
                image_stereo_coors.update({i: image_coors})

    wg = WeightedGraph()
    graph_matches = GRAPH_MATCHES
    # print('Applying weighted graph...')
    # # Apply weighted graph matching to sets of coordinates
    # for idx in image_stereo_coors.keys():
    #     print(idx/len(image_stereo_coors.keys()))
    #     c1 = image_stereo_coors.get(idx)[0]
    #     c2 = image_stereo_coors.get(idx)[1]
    #     wg.set_pairs(c1, c2)
    #     row_idx, col_idx = wg.solve()
    #     matching = [row_idx, col_idx]
    #     graph_matches.update({idx: matching})
    #     wg.clear()

    robots = {}
    corresp_coors = []
    # [(x,y),(x',y')]

    for key in graph_matches.keys():
        c1_matches = graph_matches.get(key)[0]
        c2_matches = graph_matches.get(key)[1]

        for x in c1_matches:
            for y in c2_matches:
                # from image key, get camera1 and camera2 corresponding to a match as stated
                c1_coor = image_stereo_coors.get(key)[0][x]
                c2_coor = image_stereo_coors.get(key)[1][y]
                corresp = [c1_coor, c2_coor]
                corresp_coors.append(corresp)

        print('Getting final robot for pose', key)
        num = key + 4
        name = 'frame_' + str(num) + '.png'
        final_robot = transform(corresp_coors)

        xs = [x[0] for x in final_robot]
        ys = [x[1] for x in final_robot]
        zs = [x[2] for x in final_robot]

        fig = go.Figure(data=[go.Scatter3d(x=xs, y=ys, z=zs, mode='markers')])
        # fig.show()
        print('Saving image...')
        plotly.io.write_image(fig, name)
        robots.update({key: final_robot})

    # for rob in robots.keys():
    #     # robots.keys():
    #     r = robots.get(rob)
    #     xs = [x[0] for x in r]
    #     ys = [x[1] for x in r]
    #     zs = [x[2] for x in r]
    #     fig = plt.figure(figsize=(10, 10))
    #     ax = plt.axes(projection='3d')
    #     ax.plot3D(xs, ys, zs, 'gray')
    #     ax.scatter3D(xs, ys, zs, c=zs, cmap='Greens');
    #     num = rob + 4
    #     name = 'frame_' + str(num) + '.png'
    #     ax.set_title(name)
    #     # Adjust plot view
    #     ax.view_init(elev=50, azim=225)
    #     ax.dist = 11
    #     print('saving image...')
    #     plt.savefig(name)
    #     plt.show()
    #     plt.clf()









