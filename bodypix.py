import cv2
import tensorflow as tf
from tf_bodypix.api import download_model, load_model, BodyPixModelPaths
import cv2 as cv
from matplotlib import pyplot as plt
import numpy as np

bodypix_model= load_model(download_model(BodyPixModelPaths.MOBILENET_FLOAT_50_STRIDE_16))
print(bodypix_model)


cap=cv.VideoCapture(0)
while cap.isOpened():
    ref,frame= cap.read()
    result=bodypix_model.predict_single(frame)
    mask=result.get_mask(threshold=0.5).numpy().astype(np.uint8)
    masked_image=cv2.bitwise_and(frame,frame,mask=mask)


    cv.imshow('bodypix',masked_image)


    if (cv.waitKey(10) & 0xFF==ord('q')):
        break
cap.release()
cv.destroyAllWindows()