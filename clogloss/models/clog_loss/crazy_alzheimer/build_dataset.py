import numpy as np

import cv2
import imageio
from PIL import Image


def process_one(filename):
    vid = imageio.get_reader(filename, 'ffmpeg')
    imgs = []
    for i in range(vid.count_frames()):
        frame = vid.get_data(i)
        mask = cv2.inRange(frame, (255, 69, 0), (255, 180, 0))
        mask = np.argwhere(mask)
        xmin, xmax, ymin, ymax = mask[:, 0].min(), mask[:, 0].max(
        ), mask[:, 1].min(), mask[:, 1].max()
        img = Image.fromarray(
            frame[xmin:xmax, ymin:ymax, :]).convert('L').resize((64, 64))
        imgs.append(img)
    return imgs
