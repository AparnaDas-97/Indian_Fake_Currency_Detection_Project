from Tkinter import *
import tkMessageBox
from tkFileDialog import askopenfilename
from PIL import ImageTk, Image
import cv2
import numpy as np
#import seaborn as sns
import matplotlib.pyplot as plt
from scipy import stats
import pandas as pd
import skimage
from skimage import*
from skimage.feature import greycomatrix,greycoprops
from skimage.feature import*
from sklearn import model_selection
import os
import paho.mqtt.client as mqtt
from imutils import paths
import random
from sklearn.cross_decomposition import PLSCanonical
from sklearn import svm
import pickle
from sklearn.cross_decomposition import PLSCanonical
#----------------------------------------


def training():
    img_path=sorted(list(paths.list_images("datas")))
    random.seed(42)
    random.shuffle(img_path)
    data=[]
    lbl=[]
    print(img_path)
    for i,imgs in enumerate(img_path):
        img=cv2.imread(imgs)
        img=cv2.resize(img,(500,500))

        lab_image = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
        #splitting lab image       
        l, a, b = cv2.split(lab_image)
        #applying clahe on l channel
        median = cv2.medianBlur(l,5)
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
        cl = clahe.apply(median)
        #merging l clahe and a,b channel of lab
        n=cv2.merge([cl,a,b])
        n=cv2.cvtColor(n.astype("uint8"),cv2.COLOR_LAB2BGR)
        #applying surf on merged image
        surf = cv2.xfeatures2d.SURF_create(400)
        #detecting poi in image
        kp, des = surf.detectAndCompute(n,None)
        #mark on poi 
        img2 = cv2.drawKeypoints(n,kp,None,(255,0,0),4)
        r,g,b=cv2.split(n)
        result=skimage.feature.greycomatrix(g,[6],[0, np.pi/2])
        #print np.array(img).flatten()
        data.append(np.array(img).flatten())
        l1=imgs.split(os.path.sep)[-2]
        lbl.append(int(l1))
    x=np.array(data)
    y=np.array(lbl)
    X_train, X_test, Y_train, Y_test = model_selection.train_test_split(x, y, test_size=0.25, random_state=7)
    print("OK")
    svc = svm.SVC(gamma=0.001)
    svc.fit(x,y)
    #print svc.score(X_train, Y_train)

    # save the model to disk
    filename = 'finalized_model.sav'
    pickle.dump(svc, open(filename, 'wb'))
    print("Training Completed")


    

def testing():
    root = Tk()
    root.destroy()
    path = askopenfilename()
    print(path)
    img=cv2.imread(path)
    img=cv2.resize(img,(500,500))

    lab_image = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    #splitting lab image       
    l, a, b = cv2.split(lab_image)
    #applying clahe on l channel
    median = cv2.medianBlur(l,5)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
    cl = clahe.apply(median)
    #merging l clahe and a,b channel of lab
    n=cv2.merge([cl,a,b])
    n=cv2.cvtColor(n.astype("uint8"),cv2.COLOR_LAB2BGR)
    #applying surf on merged image
    surf = cv2.xfeatures2d.SURF_create(400)
    #detecting poi in image
    kp, des = surf.detectAndCompute(n,None)
    #mark on poi 
    img2 = cv2.drawKeypoints(n,kp,None,(255,0,0),4)
    r,g,b=cv2.split(n)
    result=skimage.feature.greycomatrix(g,[6],[0, np.pi/2])
    
    loaded_model = pickle.load(open('finalized_model.sav', 'rb'))
    result= loaded_model.predict([np.array(img).flatten()])
    print(str(result[0]))
##    client = mqtt.Client()
##    client.connect("broker.hivemq.com", 1883, 60)
##    client.publish("tra",str(result[0]))

    print(result)
    
training()
testing()
