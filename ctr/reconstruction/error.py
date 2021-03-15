import numpy as np
import cv2


def error_function(x, arg1, arg2):
    """
    Applies transformation to coordinates, reflects them back to find the error.
    Takes as argument:
        x = contains the rotation and translation for camera 2
        cam1 = set of [x,y] coordinates for one image
        cam2 = set of [x,y] coordinates for second image
    """
    cam1 = arg1
    cam2 = arg2

    # Put coordinates in 2xn shape
    c1 = np.array(cam1).T
    c2 = np.array(cam2).T

    # Setting intrinsic matrix
    c2_int = np.array([[816.5455, 0, 0], [0, 816.7107, 0], [324.5504, 237.2101, 1]])
    c2_int = np.transpose(c2_int)
    c2_int = np.c_[c2_int, np.zeros(3)]
    # print(c2_int)

    # Params to optimise, using them for the extrinsic matrix
    augm = np.array([[0, 0, 0, 1]])
    c2_ex = np.reshape(x, (3, 4))
    c2_ex = np.concatenate((c2_ex, augm), axis=0)
    # print(c2_ex) reconstruction of the extrinsic matrix correct

    augm0 = np.array([[0], [0], [0]])
    c1_ex = np.concatenate((np.identity(3), augm0), axis=1)
    c1_ex = np.concatenate((c1_ex, augm), axis=0)

    # projections for camera 1 and camera 2 coordinates -> 3D
    proj1 = cv2.UMat(np.dot(c2_int, c1_ex))
    proj2 = cv2.UMat(np.dot(c2_int, c2_ex))

    projpoints1 = cv2.UMat(c1)
    projpoints2 = cv2.UMat(c2)

    # 4D real world points
    # print('triangulating...')
    points = cv2.triangulatePoints(proj1, proj2, projpoints1, projpoints2)

    # 4x n where [0] = x, 1=y, 2=z, 3=homterm
    world_coors = cv2.UMat.get(points)
    threed_hom_points = []

    a = zip(world_coors[0], world_coors[1], world_coors[2], world_coors[3])
    fourd_points = list(a)
    for p in fourd_points:
        p = np.array(p)
        x = p[0]
        y = p[1]
        z = p[2]
        aug = p[3]
        if aug == 0:
            print('dividing by 0...')
        newp = np.array([x / aug, y / aug, z / aug, 1])

        threed_hom_points.append(newp)

    err_sum = 0
    # print(len(threed_hom_points), len(c1), len(c2))

    # Find dif (ED) between projected point and world point by reprojecting back to image 2
    for i, td in enumerate(threed_hom_points):
        proj2_np = cv2.UMat.get(proj2)
        projection = np.dot(proj2_np, td)
        prj_i2 = projection / projection[2]
        prj_i2 = np.array([prj_i2[0], prj_i2[1]])
        cam2point = np.array([c2[0][i], c2[1][i]])
        err = np.linalg.norm(prj_i2 - cam2point)
        err_sum += err

    return err_sum


def objective_func(*args):
    guess, arg1, arg2 = args[0], args[1], args[2]
    perr = error_function(guess, arg1, arg2)
    return perr