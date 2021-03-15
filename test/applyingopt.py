import pandas as pd
import os
from ctr.reconstruction.error import *

dir_path = os.getcwd()
data_path = os.path.join(dir_path, "..", "data", 'man_labelled_data.csv')
df = pd.read_csv(data_path)

cam1 = df['CAM1']
cam2 = df['CAM2']

# getting array of arrays for coordinates
cam1_arr = []
for coor in cam1:
    coor = coor.split(',')
    ints = [float(x) for x in coor]
    cam1_arr.append(ints)

cam2_arr = []
for coor in cam2:
    coor = coor.split(',')
    ints = [float(x) for x in coor]
    cam2_arr.append(ints)

c2_rotation = np.transpose(np.array([[0.0218, -0.5367, 0.8435],
                                     [0.9981, 0.0607, 0.0129],
                                     [-0.0581, 0.8416, 0.5370]]))

c2_translation = np.array([[14.570], [-319.7436], [288.8639]])

# testing ....
# x = np.concatenate((c2_rotation, c2_translation), axis=1)
# x = x.flatten()

x = [687.30901993,  830.26069921,  425.03190966,   44.09333667,
       -192.52434402, -300.74397378, -553.73374624, -304.46187579,
       -246.35974378, -372.56596587, -628.65878181,  303.79493756]

err = error_function(x, cam1_arr, cam2_arr)
print('Testing... total error is...', err)

#
# # implementing optimisation
# cam1 = [[111.5963, 393.5897], [108, 328.10704067640853]]
# cam2 = [[65.8821, 36.3804], [611.2641, 271.3306]]
# x = np.concatenate((c2_rotation, c2_translation), axis=1)
# x = x.flatten()
#
# res = scipy.optimize.minimize(objective_func, x, tuple([cam1_arr, cam2_arr]))

# print(res)