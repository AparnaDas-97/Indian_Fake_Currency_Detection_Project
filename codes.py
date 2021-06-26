from Tkinter import *
import tkMessageBox
from tkFileDialog import askopenfilename
from PIL import ImageTk, Image
import cv2
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
import pandas as pd
import skimage
from skimage import*
from skimage.feature import greycomatrix,greycoprops
from skimage.feature import*
from sklearn import model_selection
import os
from imutils import paths
import random
from sklearn import svm
import pickle
from sklearn.cross_decomposition import PLSCanonical
from sklearn.metrics import confusion_matrix 
from sklearn.cross_validation import train_test_split 
from sklearn.metrics import accuracy_score 
from sklearn.metrics import classification_report
import time


t1=time.time()
img_path=sorted(list(paths.list_images("note pics")))
random.seed(42)
random.shuffle(img_path)
data=[]
lbl=[]
print img_path

X_train, X_test, Y_train, Y_test = model_selection.train_test_split(x, y, test_size=0.25, random_state=7)
print "OK"
svc = svm.SVC(kernel='linear', C=1.0)
svc.fit(x,y)
#print svc.score(X_train, Y_train)

# save the model to disk
filename = 'finalized.sav'
pickle.dump(svc, open(filename, 'wb'))
print "Completed"

try:
        for (lower, upper) in boundaries:
            # create NumPy arrays from the boundaries
            lower = np.array(lower, dtype = "uint8")
            upper = np.array(upper, dtype = "uint8")
     
            # find the colors within the specified boundaries and apply
            # the mask
            mask = cv2.inRange(image, lower, upper)
            output = cv2.bitwise_and(image, image, mask = mask)
            a=1
    except:
         pass

for i,imgs in enumerate(img_path):
    img=cv2.imread(imgs,cv2.IMREAD_GRAYSCALE)
    img=cv2.resize(img,(500,500))

    data.append(np.array(img).flatten())
    l1=imgs.split(os.path.sep)[-2]
    lbl.append(l1)
    print l1
    x=np.array(data)
    y=np.array(lbl)
y_pred = svc.predict(X_test)


print "Confusion Matrix: \n", confusion_matrix(Y_test, y_pred)
print "Accuracy :\n",accuracy_score(Y_test,y_pred)*100
print"Report : \n", classification_report(Y_test, y_pred)
t2=time.time()
print "Time  :",t2-t1


img=np.array(img).flatten()
img=np.array(img)
loaded_model = pickle.load(open('finalized.sav', 'rb'))
result= loaded_model.predict(np.array(img))

