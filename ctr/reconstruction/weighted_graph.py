from ctr.common.common import CtrClass
from ctr.resources import *
from ctr.reconstruction.error import error_function

import numpy as np
from scipy.optimize import linear_sum_assignment


class WeightedGraph(CtrClass):

    def __init__(self):
        self.cost_matrix = [[]]
        self.pairs_img1 = []
        self.pairs_img2 = []
        self.matching = []

    def set_pairs(self, pairs1, pairs2):
        self.pairs_img1 = pairs1
        self.pairs_img2 = pairs2

    def solve(self):
        self.find_cost_matrix()
        print('Finding sum assignment...')
        row_idx, col_idx = linear_sum_assignment(self.cost_matrix)
        self.matching = [row_idx, col_idx]
        return row_idx, col_idx

    def find_cost_matrix(self):
        left_img1 = self.pairs_img1
        right_img2 = self.pairs_img2

        cost_matrix = np.zeros((len(left_img1), len(right_img2)))

        # Fill cost matrix by finding error between each pair
        for i, p1 in enumerate(left_img1):
            for j, p2 in enumerate(right_img2):
                err, prj2 = error_function(BEST_TRANSF_X, p1, p2, mode='single')
                cost_matrix[i, j] = err

        self.cost_matrix = cost_matrix

    def clear(self):
        self.pairs_img1 = []
        self.pairs_img2 = []
        self.cost_matrix = [[]]
        self.matching = []




