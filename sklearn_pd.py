import  pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler,PolynomialFeatures
import matplotlib.pyplot as plt
import scipy.stats
import  numpy as np
f=open(r'E:\Iris数据集\iris.csv')
iris=pd.read_csv(f)
plt.figure(1)
iris.drop('Id',axis=1).boxplot()
std=StandardScaler(with_mean=True,with_std=True)
iris.iloc[:,1:5]=std.fit_transform(iris.iloc[:,1:5])
plt.figure(2)
iris.drop('Id',axis=1).boxplot()

le=LabelEncoder()
iris.iloc[:,5]=le.fit_transform(iris.iloc[:,5])
print(iris.iloc[:,5])
print(iris.iloc[:,5].unique())
plt.show()

np.random.seed(10)
x=scipy.stats.norm.rvs(size=100)
plt.figure
np.mean
