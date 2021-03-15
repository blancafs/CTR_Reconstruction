import os
import cv2

from ctr.common.common import CtrClass


class Reader(CtrClass):

    def read(self, path_to_cam1: str, path_to_cam2: str, path_to_back: str):
        """

        :param path_to_cam1:
        :param path_to_cam2:
        :param path_to_back:
        :return:
        """
        cam1_files = self.orderByFileName(path_to_cam1)
        cam2_files = self.orderByFileName(path_to_cam2)

        cam1_imgs = [cv2.imread(os.path.join(path_to_cam1, file)) for file in cam1_files]
        cam2_imgs = [cv2.imread(os.path.join(path_to_cam2, file)) for file in cam2_files]
        back_img = cv2.imread(os.path.join(path_to_back, 'cam1_back1.png'))

        self.info(f"reading camera data paths, such as {path_to_cam1}")

        return cam1_imgs, cam2_imgs, back_img

    def orderByFileName(self, path):
        files = os.listdir(path)
        ordered_list = []
        for name in files:
            if name.split('.')[-1] == 'png':
                number = name.split('_')[2][:-4]
                ordered_list.insert(int(number), name)
        return ordered_list
