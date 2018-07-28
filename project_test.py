import cv2
from sklearn.externals import joblib
import numpy as np
from sklearn import preprocessing

from sklearn.decomposition import PCA

clf = joblib.load("digits_cls.pkl")
#print(clf)
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

scaler = preprocessing.MinMaxScaler()

for rect in rects:
    cv2.rectangle(img,(rect[0],rect[1]),(rect[0] + rect[2],rect[1] + rect[3]),(0,255,0),3)
    leng = int(rect[3] * 1.6)
    pt1 = int(rect[1] + rect[3] // 2 - leng // 2)
    pt2 = int(rect[0] + rect[2] // 2 - leng // 2)
    roi = img_bnw[pt1:pt1+leng, pt2:pt2+leng]

    roi = cv2.resize(roi,(28,28),interpolation=cv2.INTER_AREA)
    cv2.imwrite("yo.png",roi)
    flat_data.append(roi.flatten())


flat_data = np.asarray(flat_data)

data = scaler.fit_transform(flat_data)

#pca = PCA(n_components=250)
#data = pca.fit_transform(data)
#print("PCA complete")


#stores the prediction probabilties of each character in label
#label size depends on the number of characters scanned in target and each has probability for 62 characters

label = clf.predict_proba(data)

#print classes
#print(clf.classes_)

labelX=[]

#rounding the probabilty values upto 4 places of decimal and storing in another list labelX
for i in range(len(label)):
    labelX.append([ round(elem,4) for elem in label[i] ])

top3label=[]

#sort in descending and find highest 3 probabilities
#put the char keys and respective probabilities and zip it

for i in range(len(label)):
    view=sorted(zip(clf.classes_, labelX[i]), key= lambda x: x[1], reverse=True )
    top3label.append(list(view[:3]))

#print the top 3 predictions for each character of target in terminal
for rect in rects:
    for j in range(len(top3label[0])):
        for k in range(len(top3label[0][1])):
            print(str(top3label[i][j][k]),end='  ')
    print("")
    i=i+1

#print the predictions of top 3 in a output image using cv2.putText

i=0
x=0

for rect in rects:
    y=-50
    for j in range(len(top3label[0])):
            cv2.putText(img, str(top3label[i][j][0]), (rect[0], rect[1]+y),cv2.FONT_HERSHEY_DUPLEX, 0.6, (0, 0, 255), 1)
            cv2.putText(img, str(round(top3label[i][j][1]*100,2))+"%", (rect[0]+22, rect[1]+y),cv2.FONT_HERSHEY_DUPLEX, 0.6, (0, 0, 255), 1)
            y +=20
    i += 1
cv2.imwrite("output.png",img)
