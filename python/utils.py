import os
import cv2
import numpy as np
from pathlib import Path


def show_images(*args):
    for i, image in enumerate(args):
        cv2.imshow(str(i), image)
    cv2.waitKey(-1)
    return
