from PIL import ImageTk, Image
import cv2
import numpy as np

import os
from imutils import paths
import random

#----------------------------------------


# Adding salt & pepper noise to an image
def salt_pepper(prob, img_gs):
      # Extract image dimensions
      row, col = img_gs.shape[:2]

      # Declare salt & pepper noise ratio
      s_vs_p = 0.5
      output = np.copy(img_gs)

      # Apply salt noise on each pixel individually
      num_salt = np.ceil(prob * img_gs.size * s_vs_p)
      coords = [np.random.randint(0, i - 1, int(num_salt))
            for i in img_gs.shape]
      output[coords] = 1

      # Apply pepper noise on each pixel individually
      num_pepper = np.ceil(prob * img_gs.size * (1. - s_vs_p))
      coords = [np.random.randint(0, i - 1, int(num_pepper))
            for i in img_gs.shape]
      output[coords] = 0

      return output


def training():
    img_path=sorted(list(paths.list_images("datas")))
    random.seed(42)
    random.shuffle(img_path)
    data=[]
    lbl=[]
    print (len(img_path))
    a = 0
    for i,imgs in enumerate(img_path):
        img=cv2.imread(imgs)
        img=cv2.resize(img,(500,500))
        edges = cv2.Canny(img,100,200)


        ret,thresh1 = cv2.threshold(img,127,255,cv2.THRESH_BINARY)
        ret,thresh2 = cv2.threshold(img,127,255,cv2.THRESH_BINARY_INV)
        ret,thresh3 = cv2.threshold(img,127,255,cv2.THRESH_TRUNC)
        ret,thresh4 = cv2.threshold(img,127,255,cv2.THRESH_TOZERO)
        ret,thresh5 = cv2.threshold(img,127,255,cv2.THRESH_TOZERO_INV)

        lab_image = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
        kernel = np.ones((5,5),np.float32)/25
        dst = cv2.filter2D(img,-1,kernel)
        #splitting lab image       
        l, a, b = cv2.split(lab_image)
        #applying clahe on l channel
        median = cv2.medianBlur(l,5)
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
        cl = clahe.apply(median)
        #merging l clahe and a,b channel of lab
        n=cv2.merge([cl,a,b])
        sp_05 = salt_pepper(0.5, img)
        target_dir = 'pre'
        if not os.path.exists(target_dir):
          os.mkdir(target_dir)
        # Store the resultant image as 'sp_05.jpg'
        cv2.imwrite('pre/p_'+str(a)+'.jpg', img)
        a+=1
training()
