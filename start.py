import numpy as np
import matplotlib.pyplot as pt
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
data=pd.read_csv("datasets\train.csv").as_matrix()
clf=DecisionTreeClassifier()

#training dataset
xtrain = data[0:29000,1:]
train_label=data[0:29000,0]

clf.fit(xtrain,train_label)

#testing data
xtest=data[13000:,1:1]
actual_label=data[13000:,0]

d=xtest[8]
d.shape=(28,28)
pt.imshow(255-d,cmap='gray')
print(clf.predict([xtest[8]]))
pt.show()

p=clf.predict(xtest)

count=0
for i in range(0,13000):
    count+=1 if p[i]==actual_label[i] else 0
print("accuracy=",(count/13000)*100)
