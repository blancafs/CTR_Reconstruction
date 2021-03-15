import os

# cur path to ctr
CTR_FOLDER = os.path.dirname(os.path.realpath(__file__))
ROOT_FOLDER = os.path.join(CTR_FOLDER, '..')
DATA_FOLDER = os.path.join(ROOT_FOLDER, 'test', 'data')
CAM1_FOLDER = os.path.join(DATA_FOLDER, 'cam1')
CAM2_FOLDER = os.path.join(DATA_FOLDER, 'cam2')
BACKG_FOLDER = os.path.join(DATA_FOLDER, 'background')
LF_RESULTS_FOLDER = os.path.join(ROOT_FOLDER, 'data', 'linefit_results')