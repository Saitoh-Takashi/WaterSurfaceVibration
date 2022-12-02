import cv2
import numpy as np
from dataclasses import dataclass


@dataclass
class ImageArea:
    x1: int
    y1: int
    x2: int
    y2: int

    def clip(self, img: np.ndarray) -> np.ndarray:
        return img[self.y1:self.y2 + 1, self.x1: self.x2 + 1]


def water_surface_height(img: np.ndarray) -> np.float64:
    surface_points = np.where(img != 0)
    return np.average(surface_points[0])


def main():
    roi = ImageArea(x1=525, y1=800, x2=555, y2=1000)
    cap = cv2.VideoCapture('C:/video/02.mp4')

    frame_index = 0
    while True:
        ret, img = cap.read()

        if ret:
            img = roi.clip(img)
            gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            canny = cv2.Canny(gray_img, threshold1=0, threshold2=80)

            cv2.imwrite(f'pic/{frame_index:0=5}.jpg', canny)

            frame_index += 1

        else:
            break


if __name__ == '__main__':
    main()
