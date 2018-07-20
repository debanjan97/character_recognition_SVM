import cv2
from sklearn.externals import joblib
import numpy as np
from sklearn import preprocessing
from sklearn.decomposition import PCA

clf = joblib.load("digits_cls.pkl")
flat_data = []

img = cv2.imread("target.jpeg")

img_g = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
img_g = cv2.GaussianBlur(img_g,(5,5),0)

ret,img_bnw = cv2.threshold(img_g,90,255,cv2.THRESH_BINARY_INV)
print(ret)

_,ctrs,_ = cv2.findContours(img_bnw.copy(),cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
cv2.drawContours(img_bnw, ctrs, -1, (0,255,0), 3)

rects = []
#img_g = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
for ctr in ctrs:
    rects.append(cv2.boundingRect(ctr))

cv2.imwrite("target_example_bnw.png",img_bnw)

#print(rects)
scaler = preprocessing.MinMaxScaler()
for rect in rects:
    #data = flat_data
    cv2.rectangle(img,(rect[0],rect[1]),(rect[0] + rect[2],rect[1] + rect[3]),(0,255,0),3)
    leng = int(rect[3] * 1.6)
    pt1 = int(rect[1] + rect[3] // 2 - leng // 2)
    pt2 = int(rect[0] + rect[2] // 2 - leng // 2)
    roi = img_bnw[pt1:pt1+leng, pt2:pt2+leng]

    roi = cv2.resize(roi,(28,28),interpolation=cv2.INTER_AREA)
#    roi = np.asarray(roi)
    cv2.imwrite("yo.png",roi)
    flat_data.append(roi.flatten())


flat_data = np.asarray(flat_data)

data = scaler.fit_transform(flat_data)

#pca = PCA(n_components=250)
#data = pca.fit_transform(data)
#print("PCA complete")

i = 0
#data = data[-len(rects):]
#print(data)
label = clf.predict(data)
for rect in rects:
    cv2.putText(img, str(label[i]), (rect[0], rect[1]),cv2.FONT_HERSHEY_DUPLEX, 2, (255, 255, 255), 3)
    i += 1



cv2.imwrite("target_example.png",img)
