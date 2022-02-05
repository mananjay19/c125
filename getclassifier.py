import cv2 
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from PIL import Image
import PIL.ImageOps

X,y=fetch_openml('mnist_784',version=1,return_X_y=True)
#print(pd.Series(y).value_counts())
classes=['0','1','2','3','4','5','6','7','8','9']
nclasses=len(classes)

xtrain,xtest,ytrain,ytest=train_test_split(X,y,test_size=2500,random_state=0,train_size=7500)
xtrainscale=xtrain/255.0
xtestscale=xtest/255.0
clf=LogisticRegression(solver='saga',multi_class='multinomial').fit(xtrainscale,ytrain)
#ypred=clf.predict(xtestscale)
#accuracy=accuracy_score(ytest,ypred)
#print(accuracy)
def getprediction (image):
    impil=Image.open(image)
    imagebw=impil.convert('L')
    imagebwresize=imagebw.resize((28,28),Image.ANITIALIAS)
    pixelfilter=20
    minpixel=np.percentile(imagebwresize,pixelfilter)
    imagebwresizescale = np.clip(imagebwresize-minpixel,0,255)
    maxpixel=np.max(imagebwresize)
    imagebwresizescale=np.asarray(imagebwresizescale)/maxpixel
    testsample=np.array(imagebwresizescale).reshape(1,784)
    testpred=clf.predict(testsample)
    return testpred[0]