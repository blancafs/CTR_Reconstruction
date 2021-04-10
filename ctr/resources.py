import os
import pickle

# cur path to ctr
CTR_FOLDER = os.path.dirname(os.path.realpath(__file__))
ROOT_FOLDER = os.path.join(CTR_FOLDER, '..')
DATA_FOLDER = os.path.join(ROOT_FOLDER, 'test', 'data')
TEST_FOLDER = os.path.join(ROOT_FOLDER, 'test')
CAM1_FOLDER = os.path.join(DATA_FOLDER, 'cam1')
CAM2_FOLDER = os.path.join(DATA_FOLDER, 'cam2')
BACKG_FOLDER = os.path.join(DATA_FOLDER, 'background')
LF_RESULTS_FOLDER = os.path.join(ROOT_FOLDER, 'data', 'linefit_results')
BEST_TRANSF_X = [687.30901993, 830.26069921, 425.03190966, 44.09333667,
                 -192.52434402, -300.74397378, -553.73374624, -304.46187579,
                 -246.35974378, -372.56596587, -628.65878181, 303.79493756]

# TRUTH DATA
c1 = os.path.join(TEST_FOLDER, 'cam1t.pkl')
c2 = os.path.join(TEST_FOLDER, 'cam2t.pkl')
c1_file = open(c1, "rb")
TRUTH_CAM1 = pickle.load(c1_file)
c2_file = open(c2, "rb")
TRUTH_CAM2 = pickle.load(c2_file)