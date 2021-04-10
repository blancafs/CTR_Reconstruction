import matplotlib.pyplot as plt

def show_robot_3d(coors_dict):
    # coors in format [i:[x, y, z]...]
    for rob in coors_dict.keys():
        r = coors_dict.get(rob)
        xs = [x[0] for x in r]
        ys = [x[1] for x in r]
        zs = [x[2] for x in r]
        # print('[SHOWROBOT3D]:showing figure...')
        fig = plt.figure(figsize=(10, 10))
        ax = plt.axes(projection='3d')
        ax.plot3D(xs, ys, zs, 'gray')
        ax.scatter3D(xs, ys, zs, c=zs, cmap='Greens')

        name = 'robot_' + str(rob) + '.png'
        ax.set_title(name)
        plt.show()


def show_reflection(image, c2coors, c):
    print('showing reflection .. ')
    # coors in format [i:[x, y, z]...]
    xs = [x[0] for x in c2coors]
    ys = [x[1] for x in c2coors]

    plt.scatter(xs, ys)
    plt.imshow(image)

    title='cam2_'+str(c)+'.png'
    plt.title(title)
    plt.savefig(title)
    plt.show()
    # print('showing figure...')


def join_corresp_coors(good_coor_sets, graph_matches):
    # corresp coors in format [(x,y)(x',y')]
    final_corresp = {}
    # graphmatches is a dict in format idx: c1, c2
    for key in graph_matches.keys():
        c1_matches = graph_matches.get(key)[0]
        c2_matches = graph_matches.get(key)[1]
        corresp_coors = []
        zipped = list(zip(c1_matches, c2_matches))

        for x, y in zipped:
            # from image key, get camera1 and camera2 corresponding to a match as stated
            c1_coor = good_coor_sets.get(key)[0][x]
            c2_coor = good_coor_sets.get(key)[1][y]
            corresp = [c1_coor, c2_coor]
            corresp_coors.append(corresp)
        final_corresp.update({key: corresp_coors})

    return final_corresp
#
