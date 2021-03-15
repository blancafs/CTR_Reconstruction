import csv
import pandas as pd
import numpy as np
import statistics
import matplotlib.pyplot as plt
from ast import literal_eval

# def calculate_3d_distance(points1, points2):
#     x1 = points1[0]
#     x2 = points2[0]
#     y1 = points1[1]
#     y2 = points2[1]
#     z1 = points1[2]
#     z2 = points2[2]
#     sq_sum = np.square((x1 - x2)) + np.square((y1 - y2)) + np.square((z1 - z2))
#     dis = np.sqrt(sq_sum)
#     return dis
#
#
# back = pd.read_csv('csvs/transformed_back.csv')
# edge = pd.read_csv('csvs/transformed_edge.csv')
#
# dis_differences = []
#
# back_readings = back['3dcoors']
# arr_readings = []
# tip_readings = back['truth']
# arr_tips = []
#
# for b in back_readings:
#     coors = [float(x) for x in b.split(' ')]
#     arr_readings.append(coors)
#
# for b in tip_readings:
#     coors = [float(x) for x in b.split(' ')]
#     arr_tips.append(coors)
#
# for x in zip(arr_readings, arr_tips):
#     dis = calculate_3d_distance(x[0], x[1])
#     dis_differences.append(dis)
#
# print(dis_differences)

BACK = [4.718007313474616, 2.761039175745273, 1.5572227265230505, 8.182750750367777, 4.444786783412683, 2.347886375019049, 2.761039175745273,1.761039175745273,2.896853370469421, 1.353706844187458, 2.9246262769796614, 4.730187415948742,2.3246262769796614]
EDGE = [54.21826627668945, 44.58088366430167, 20.913931192628517, 25.6, 31.79150205023349, 39.099309301955714, 9.77284606754859, 18, 27.5800424816569, 20.913931192628517, 52.86461913709395, 8.28592846336486,20.913931192628517, 47.49709403858303, 18, 29.201883183452377, 16.43510308425232]

BACK = [(x-(0.01*x))/100 for x in BACK]
EDGE = [(x-(0.01*x))/100 for x in EDGE]

curr = BACK
mean = statistics.mean(curr)
std = statistics.stdev(curr)
err = [(x-mean) for x in curr]
means = [mean] * len(curr)
highs = [(mean+e) for e in err]
lows = [(mean-e) for e in err]

plt.boxplot(curr)
# plt.plot(range(len(curr)), means)
# plt.errorbar(range(len(curr)), y=ys, yerr=err, capsize=4)
#plt.errorbar(range(len(means)), means, yerr=stds)
#plt.plot(range(len(means)), means)
# plt.fill_between(range(len(curr)), lows, highs, alpha=0.2)
plt.title('Background Subtraction: 3D error')
plt.xticks([])
plt.ylabel('Distance in cm')
plt.show()