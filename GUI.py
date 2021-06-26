from tkinter import*
import sys
from tkinter.filedialog import askopenfile

import cv2
import os
import numpy as np
from PIL import Image,ImageTk
import random
import _thread
import time
import pandas as pd
import numpy as np
import os, cv2
import glob
import matplotlib.pyplot as plt
from tkinter import filedialog
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
from imutils import paths
import random
from sklearn.cross_decomposition import PLSCanonical
from sklearn import svm
import pickle
from sklearn.neighbors import KNeighborsClassifier
def pp(a):
    pass
    #global mylist
    #mylist.insert(END, a)

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
    print ("OK")
    svc = svm.SVC(kernel='linear', C=1.0)
    svc.fit(x,y)
    knn = KNeighborsClassifier(n_neighbors=3) 
  
    knn.fit(x,y) 
    #print svc.score(X_train, Y_train)

    # save the model to disk
    filename = 'finalized_modelsvm.sav'
    pickle.dump(svc, open(filename, 'wb'))
    filename = 'finalized_modelknn.sav'
    pickle.dump(knn, open(filename, 'wb'))
    print ("Training Completed")
    print ("Accuracy on training set:")
    print (knn.score(X_train, Y_train))
    print ("Accuracy on testing set:")
    print (knn.score(X_test, Y_test))


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
    X_train, X_test, Y_train, Y_test = model_selection.train_test_split(x, y, test_size=0.25, random_state=7)
    print ("OK")
    svc = svm.SVC(kernel='linear', C=1.0)
    svc.fit(x,y)
    knn = KNeighborsClassifier(n_neighbors=3) 
  
    knn.fit(x,y) 
    #print svc.score(X_train, Y_train)

    # save the model to disk
    filename = 'finalized_modelsvm.sav'
    pickle.dump(svc, open(filename, 'wb'))
    filename = 'finalized_modelknn.sav'
    pickle.dump(knn, open(filename, 'wb'))
    print ("Training Completed")
    print ("Accuracy on training set:")
    print (knn.score(X_train, Y_train))
    print ("Accuracy on testing set:")
    print (knn.score(X_test, Y_test))
    root.after(500, lambda : pp("Data Set Loaded Properly."))
    
    root.after(1000, lambda : pp("All the datas are processing."))
    
    root.after(3500, lambda : pp("Training started."))

    root.after(3500, lambda : pp("Model Saved."))
    
    root.after(5500, lambda : pp("________________________________________"))
    
    

def testing():
    global lbl, txtdisplay,img
    #path =  filedialog.askopenfilename(initialdir = "E:/Projects 2020/",title = "choose your file")
    #print (path)
    #txtdisplay.delete("1.0", "end")
    #txtdisplay.insert(END, path)
    #cimg=cv2.imread(path)
    # resize the image
    img = cv2.resize(img, (500, 500))
    #img = Image.open(filename)



    #cv2.imshow("currency",img)
    #canvas.create_image(20, 20, anchor=NW, image=img)




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

    #canvas = Canvas(root, width=300, height=300)
    #canvas.pack()
    #img = ImageTk.PhotoImage(img)
    #canvas.create_image(20, 20, anchor=NW, image=img)
    #canvas.place(x=500, y=450)

    loaded_model = pickle.load(open('finalized_modelsvm.sav', 'rb'))
    result= loaded_model.predict([np.array(img).flatten()])
    print (str(result[0]))
    if result[0]==0:
        result="2000 Rupee Real Note"
    if result[0]==1:
        result="2000 Rupee Fake Note"
    if result[0]==2:
        result="100 Rupee Real Note"
    if result[0]==3:
        result="100 Rupee Fake Note"
    if result[0]==4:
        result="200 Rupee Real Note"
    if result[0]==5:
        result="200 Rupee Fake Note"
    if result[0]==6:
        result="500 Rupee Real Note"
    if result[0]==7:
        result="500 Rupee Fake Note"
    

    print (result)
    lbl.delete("1.0", "end")
    lbl.insert(END, result)
    root.after(500, lambda : pp("Image Loaded Properly."))
    
    root.after(1000, lambda : pp("Image processing."))
    
    root.after(3500, lambda : pp("Prediction started."))

    root.after(3500, lambda : pp("Result: "+result))
    
    root.after(5500, lambda : pp("________________________________________"))

    cv2.waitKey(10000)

    panel.destroy()

    return img
    
    
    
##train_new()
##testing("5imagef.jpg")





    

root = Tk()
root.geometry("1600x700+0+0")
root.title("Fake Currency Detection")

image = Image.open("bg.jpg")
image = image.resize((1600, 700), Image.ANTIALIAS) ## The (550, 250) is (height, width)
pic = ImageTk.PhotoImage(image)
lbl_reg=Label(root,image=pic,anchor=CENTER)
lbl_reg.place(x=0,y=0)
#------------------TIME--------------
localtime=time.asctime(time.localtime(time.time()))
#-----------------INFO TOP------------
lblinfo = Label(root, font=( 'aria' ,30, 'bold' ),text="Fake Currency Detection",bg="white",fg="black",bd=10,anchor='w')
lblinfo.place(x=400,y=10)
lblinfo = Label(root, font=( 'aria' ,20, ),text=localtime,bg="white",fg="black",anchor=W)
lblinfo.place(x=450,y=80)

#lblin = Label(root, font=( 'aria' ,15, 'bold' ),text="Train with Dataset",bg="white",fg="black",bd=10,anchor='w')
#lblin.place(x=140,y=220)

#btntrn=Button(root,padx=16,pady=8, bd=10 ,bg="#3dbaea",fg="black",font=('ariel' ,16,'bold'),width=10, text="TRAIN", command=lambda:train_new())
#btntrn.place(x=150, y=260)



#lbli = Label(root, font=( 'aria' ,15, 'bold' ),text="Test Your Currency",bg="white",fg="black",bd=10,anchor='w')
#lbli.place(x=150,y=310)

#txtdisplay = Entry(root,font=('ariel' ,20,'bold') , bd=5 ,insertwidth=7 ,bg="white",justify='right')
#txtdisplay.place(x=80,y=350)

#


image11 = Image.open("up.png")
image12 = image11.resize((20, 20), Image.ANTIALIAS) ## The (550, 250) is (height, width)
pic12 = ImageTk.PhotoImage(image12)

btntrn=Button(root,padx=16,pady=8, bd=10 ,bg="#3dbaea",fg="black",font=('ariel' ,16,'bold'),width=12, text = 'Predict', command=lambda: testing())
btntrn.place(x=150, y=300)







def openfn():
    global filename
    #filename = filedialog.askopenfilename(title='open')
    filename = filedialog.askopenfilename(title='open')
    return filename

def open_img():
    #global cimg
    #x= openfn()
    #print(img)
    #img = Image.open(x)
    #img = img.resize((500,300), Image.ANTIALIAS)
    #img = ImageTk.PhotoImage(img)
    #panel = Label(root, image=img)
    #panel.image = img
    #panel.place(x=150,y=400)
    #panel.pack(padx=20, pady=20, side=RIGHT)
    global img,panel
    x1 = []
    path = askopenfile()
    n = path.name
    img = cv2.imread(n)
    shw=Image.open(n)
    #shw = img.copy()
    shw = shw.resize((500,300),Image.ANTIALIAS)
    shw = ImageTk.PhotoImage(shw)
    panel = Label(root, image=shw)
    panel.image = shw
    panel.pack(padx=20, pady=20, side=RIGHT)

    #cv2.imshow("Selected currency", shw)
    #cv2.waitKey(0)




btntrn=Button(root,padx=16,pady=8, bd=10 ,bg="#3dbaea",fg="black",font=('ariel' ,16,'bold'),width=12, text = 'Test Your Currency', command=lambda: open_img())
btntrn.place(x=150, y=160)











lbl = Text(root, font=( 'aria' ,20, 'bold' ),bg="white",fg="black",bd=10, height=4, width=50)
lbl.place(x=140,y=500)

def qexit():
    root.destroy()



#mylist = Listbox(root,width=70, height=20, bg="#75eeff" )

#mylist.place( x = 880, y = 100 )



btnexit=Button(root,padx=16,pady=8, bd=10 ,bg="#3dbaea",fg="black",font=('ariel' ,16,'bold'),width=10, text="EXIT",command=qexit)
btnexit.place(x=1000, y=500)






root.mainloop()

