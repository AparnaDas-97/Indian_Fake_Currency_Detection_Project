from tkinter import*
import sys
from tkinter.filedialog import askopenfile

import cv2
import os
import numpy as np
import PIL
import random
import _thread
import time
import pandas as pd
import numpy as np
import os, cv2
import glob
import matplotlib.pyplot as plt
from tkinter import filedialog
import PIL.Image
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
from imutils import paths
import random
from sklearn.datasets import fetch_openml
from sklearn.cross_decomposition import PLSCanonical
from sklearn import svm
import pickle
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix 
from sklearn.model_selection import train_test_split 
from sklearn.metrics import accuracy_score 
from sklearn.metrics import classification_report
from sklearn.metrics import average_precision_score
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import plot_precision_recall_curve
import matplotlib.pyplot as plt
from sklearn.metrics import precision_recall_curve, roc_curve
from sklearn.preprocessing import label_binarize


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

   



def train_new():
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
        data.append(np.array(img).flatten())
        ll=imgs.split(os.path.sep)[-2]
        lbl.append(int(ll))
        print(ll)
        a+=1
    x=np.array(data)
    y=np.array(lbl)

    #Y = label_binarize(x, classes=[*range(8)])

    #X_train, X_test, y_train, y_test = train_test_split(mnist.data,Y,random_state = 42)
    X_train, X_test, Y_train, Y_test = model_selection.train_test_split(x, y, test_size=0.3, random_state=7)
    print ("OK")
    #svc = svm.SVC(kernel='linear', C=1.0)
    #svc.fit(x,y)
    knn = KNeighborsClassifier(n_neighbors=3) 
  
    knn.fit(x,y) 
    #print svc.score(X_train, Y_train)

    # save the model to disk
    #filename = 'finalized_modelsvm.sav'
    #pickle.dump(svc, open(filename, 'wb'))
    filename = 'finalized_modelknn.sav'
    pickle.dump(knn, open(filename, 'wb'))
    print ("Training Completed")
    print ("Accuracy on training set:")
    print (knn.score(X_train, Y_train))
    print ("Accuracy on testing set:")
    print (knn.score(X_test, Y_test))

    '''print ("Accuracy on training set:")
    print (svc.score(X_train, Y_train))
    print ("Accuracy on testing set:")
    print (svc.score(X_test, Y_test))'''

    y_pred = knn.predict(X_test)
    #n_classes = 8

    print ("Confusion Matrix: \n", confusion_matrix(Y_test, y_pred))
    print ("Accuracy :\n",accuracy_score(Y_test,y_pred)*100)
    print("Report : \n", classification_report(Y_test, y_pred))
    
    y_score = knn.predict_proba(X_test)

    '''precision = dict()
    recall = dict()
    for i in range(8):
        precision[i], recall[i], _ = precision_recall_curve(Y_test[:, i],
                                                        y_score[:, i])
        plt.plot(recall[i], precision[i], lw=2, label='class {}'.format(i))
    
    plt.xlabel("recall")
    plt.ylabel("precision")
    plt.legend(loc="best")
    plt.title("precision vs. recall curve")
    plt.show()'''
   








    

root = Tk()
root.geometry("1600x700+0+0")
root.title("Fake Currency Detection")

btntrn=Button(root,padx=16,pady=8, bd=10 ,bg="#3dbaea",fg="black",font=('ariel' ,16,'bold'),width=10, text="TRAIN", command=lambda:train_new())
btntrn.place(relx=0.5, rely=0.5, anchor=CENTER)














root.mainloop()

