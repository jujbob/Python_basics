import math
import numpy as np
# import pandas as pd
import cv2
import os


def print_hi(name):
  print(f'Hi,{name}')

def mouse_handler(event,x,y,flags,param):

    if event == cv2.EVENT_LBUTTONUP:
        img = ori_img.copy()

        src.append([x,y])

        for xx, yy in src:
            cv2.circle(img, center = (xx, yy), radius=5, color=(0, 255, 0), thickness= -1, lineType = cv2.LINE_AA)
        
        cv2.imshow('img', img)

        if len(src) == 4:
            src_np = np.array(src, dtype=np.float32)

            width = max(np.linalg.norm(src_np[0] - src_np[1]), np.linalg.norm(src_np[2] - src_np[3]))
            height = max(np.linalg.norm(src_np[0] - src_np[3]), np.linalg.norm(src_np[1] - src_np[2]))

            width = math.floor(width)
            height = math.floor(height)
            
            dst_np = np.array([
                [0,0],
                [width, 0],
                [width, height],
                [0, height]        
            ], dtype=np.float32)

            M = cv2.getPerspectiveTransform(src=src_np, dst=dst_np)
            result = cv2.warpPerspective(ori_img, M=M, dsize=(width, height))
            img_numpy = np.array(result, 'uint8')
            result = cv2.cvtColor(img_numpy, cv2.COLOR_BGR2GRAY)
            # result = img_numpy
            # result = cv2.adaptiveThreshold(result, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)

            cv2.imshow('result', result)
            cv2.imwrite('./result/%s_result%s' % (filename,ext), result)



# Press the green button in the gutter to run the script
if __name__ == '__main__':
    print_hi('ktlim')
    img_path = './img/cat.jpeg'
    filename, ext = os.path.splitext(os.path.basename(img_path))

    ori_img = cv2.imread(img_path)

    src = []
    cv2.namedWindow('img')
    cv2.setMouseCallback('img', mouse_handler)

    cv2.imshow("img", ori_img)
    cv2.waitKey(0)