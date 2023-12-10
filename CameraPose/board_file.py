
import numpy as np
import json
import os

class BoardDefinition:
    def __init__(self):
        self.file_location = os.path.join(os.path.split(os.path.dirname(__file__))[0], 'config', 'board.json')
        self.data = self.load_board()
        self.mm_to_px = self.find_mm_to_px()
        self.marker_loaction_in_mm = None
        self.add_markers()
        self.marker_size_mm = self.get_marker_size()

    def load_board(self):
        with open(self.file_location) as f:
            data = json.load(f)
        return data

    def find_mm_to_px(self):
        paper_size_mm = self.data['paper_size_mm']
        dpi = self.data['DPI']
        paper_in_px = paper_size_mm[0] * dpi / 25.4
        mm_to_px = paper_in_px / paper_size_mm[0]
        return mm_to_px

    def px_to_mm(self, px):
        return px / self.mm_to_px

    def centeroidnp(self,arr):
        length = arr.shape[0]
        sum_x = -np.sum(arr[:, 0])
        sum_y = np.sum(arr[:, 1])
        sum_z = np.sum(arr[:, 2])
        return np.array([sum_x/length, sum_y/length, sum_z/length])

    def add_markers(self):
        self.marker_loaction_in_mm = {}
        for pts in self.data['points']:
            arr = np.array(pts['pts'])
            centroid = self.centeroidnp(arr)
            centroid_in_px = centroid / self.mm_to_px
            self.marker_loaction_in_mm[pts['id']] = centroid_in_px

    def get_marker_size(self):
        marker_size_px = self.data['tag_size_px'] / self.mm_to_px
        return marker_size_px


    def board_center(self):
        # return the center of the board in mm
        paper_size_mm = self.data['paper_size_mm']
        margin_size = self.data["margins_mm"]
        tag_mid_mm =   self.data["tag_size_mm"]/2

        return np.array([paper_size_mm[0]/2 - margin_size - tag_mid_mm, paper_size_mm[1]/2 - margin_size - tag_mid_mm, 0])

# example run
# if __name__ == '__main__':
#     bd = BoardDefinition()
#     print(bd.marker_loaction_in_mm)


