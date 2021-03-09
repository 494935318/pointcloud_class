import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
#matplotlib inline
from matplotlib.font_manager import FontProperties
fonts=FontProperties(fname='/Library/Fonts/华文细黑。ttf',size=14)
from sklearn import metrics
from sklearn.model_selection import  train_test_split
from sklearn.datasets import  load_iris
iris=load_iris()
#K折交叉
from sklearn.model_selection import  KFold
from sklearn.model_selection import  cross_val_score
from sklearn.model_selection import  cross_val_predict
#线性判别分类器
from sklearn.discriminant_analysis import  LinearDiscriminantAnalysis,LinearClassifierMixin
#k近邻
from sklearn.neighbors import KNeighborsClassifier
irkf=KFold(n_splits=10,random_state=2)
LDA_clf=LinearDiscriminantAnalysis(n_components=2)
score=[]
for ii,(train_index,text_index) in enumerate(irkf.split(iris.data)):
    LDA_clf.fit(iris.data[train_index],iris.target[train_index])
    prey=LDA_clf.predict(iris.data[text_index])
    acc=metrics.accuracy_score(iris.target[text_index],prey)#正确率
    print("Fold:",ii+1,'Acc:',np.round(acc,4))
    score.append(acc)
print('ave:',np.mean(score))

scores1 = cross_val_score(LDA_clf, iris.data[train_index],iris.target[train_index],cv=10)
print(np.mean(scores1))
kn=KNeighborsClassifier(n_neighbors=10)
scores2=cross_val_score(kn,iris.data[train_index],iris.target[train_index],cv=10)
print(np.mean(scores2))

#分层交叉验证
from sklearn.model_selection import  StratifiedKFold
skf_ir=StratifiedKFold(n_splits=3,random_state=2)
scores3=[]
for ii ,(train_index,text_index) in enumerate(skf_ir.split(iris.data,iris.target)):
    LDA_clf.fit(iris.data[train_index], iris.target[train_index])
    prey = LDA_clf.predict(iris.data[text_index])
    acc = metrics.accuracy_score(iris.target[text_index], prey)  # 正确率
    print("Fold:", ii + 1, 'Acc:', np.round(acc, 4))
    scores3.append(acc)
print('ave:', np.mean(scores3))
