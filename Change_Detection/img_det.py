import argparse
import cv2
import time

from typing import Any


def main(args: Any) -> None:
    img1 = cv2.imread("G:\demo_code\move_obj_det\img\lQLPJxGrrLPfzZbNBDjNB4Cwo5Ph8EzmbzcEItShnUDLAA_1920_1080.png")
    img2 = cv2.imread("G:\demo_code\move_obj_det\img\lQLPJxwCTeYhjpbNBDjNB4CwTTvcvOYmg8sEItShnUAuAA_1920_1080.png")

    threshold: int = args.threshold

    bg = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)

    # gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    gray = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

    mask = cv2.absdiff(gray, bg)

    mask[mask < threshold] = 0
    mask[mask >= threshold] = 255

    cv2.imwrite("mask.png",mask)


    cv2.imshow('Mask', mask)
    cv2.imshow('Frame', gray)
    cv2.imshow('Background', bg)
    cv2.waitKey(0)

    # cap.release()
    # cv2.destroyAllWindows()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Sample script of inter-frame defference detection.')
    # _ = parser.add_argument('--camera', '-c', type=int, default=0, help='camera device number, by default 0.')
    _ = parser.add_argument('--camera', '-c', default="video/20230316150539.mp4", help='camera device number, by default 0.')
    _ = parser.add_argument('--threshold', '-t', type=int, default=30,
                            help='threshold of masking image, by default 30.')
    _ = parser.add_argument('--refresh', '-r', type=int, default=10, help='refresh count of base image, by default 30.')
    args = parser.parse_args()
    main(args)