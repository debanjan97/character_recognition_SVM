#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import time
import cv2
import os
import glob
import numpy as np
import csv
from sklearn import model_selection, preprocessing, neighbors,svm
from sklearn.externals import joblib
from sklearn.decomposition import PCA

#please change image_dir to the directory you have saved dope in

img_dir = "/home/srinjoy/character_recognition_SVM/dope"
data_path = os.path.join(img_dir,'*g')

files = glob.glob(data_path)
#print(type(files))
files.sort()

#magic defines how many images in the datasets are used for training
#magic=2 means that every one of 2 images are taken
magic = 2

img = np.array([])
#label = label.flatten()
print("Labels have been generated !")
data = []
i = 1
for f in files:
	if i%magic == 0:
#		print(f)
		img = cv2.imread(f)
		img = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
		img = cv2.resize(img,(28,28))
		_,img = cv2.threshold(img,90,255,cv2.THRESH_BINARY_INV)
		if i == 5220:
			cv2.imwrite('sample_mod.png',img)
		print("image %d has been read"%i)
		data.append(img)
		#os.system("cp %s /home/eyexore/work/ardent/dope_test/%c.png"%(f,label[i]))
	i += 1

#prints the number of images used to train our model
print("Number of Images = %d"%(i/magic))
i = 1
label = []

#the csv file holds the labels of each 1071 examples sampled of each character
with open('label_dope.csv','rt') as label_file:
	reader = csv.reader(label_file)
	for row in reader:
		if(i%magic == 0):
			label.append(row)
		i +=1


data,label = np.asarray(data),np.asarray(label)

flat_data = []
for i in range(len(data)):
	flat_data.append(data[i].flatten())

#joblib.dump(flat_data,"data.pkl",compress=3)

flat_data = np.asarray(flat_data)
label = label.flatten()
scaler = preprocessing.MinMaxScaler()
data = scaler.fit_transform(flat_data)

#pca = PCA(n_components=250)

#data = pca.fit_transform(data)
#print("PCA done")

#we define the training and testing set
X_train, X_test, y_train, y_test = model_selection.train_test_split(data,label,test_size=0.3)

#using SVM and probabilty value set to true to get corresponding probabilty values as well
clf = svm.SVC(gamma=0.001,probability=True)

clf.fit(X_train,y_train)

#print the accuracy score of our model
#print("Accuracy is : {}".format(clf.score(X_test,y_test)))

print(clf.score(X_test,y_test))

#dump the trained mdoel in a pickle file to be used later 
joblib.dump(clf, "digits_cls.pkl", compress=3)
