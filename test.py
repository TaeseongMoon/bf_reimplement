
import os
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"   
os.environ["CUDA_VISIBLE_DEVICES"]="0"
from blazeface import blazeface
from utils import draw_landmark_on_image
import cv2
@profile
def main():
    img = cv2.imread("test.jpg")
    bf = blazeface()
    landmark, bbox = bf.detect(img)
    img = draw_landmark_on_image(img, landmark, bbox)
    cv2.imwrite('dasd.jpg', img)

if __name__ == "__main__":
    main
