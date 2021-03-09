import statsmodels.api as sm
import  statsmodels.formula.api as smf
import  pandas as pd
import numpy as np
import  os
a=np.array(np.arange(10)).reshape(2,5)
b=np.array(np.arange(10)).reshape(2,5)
c=np.concatenate((a,b),axis=0)#合并矩阵，指定维度

print(c)
d,e,f=np.split(c,[2,4],1)#切片指定维度，和位置切片
print(d,'\n',e,'\n',f)
z=np.zeros((1,2,3))
np.random.seed(1)
print(np.random.randn(3,3))#标准正态分布
print(np.random.rand(3,3))#0-1随机数
b=np.array([[1,2,3],[2,3,4]])#生成矩阵
c=np.zeros((3,2,1))
np.squeeze(c)#删除值为1的维度
np.argmax(b,1)#返回指定维度最大值的index,返回矩阵指定维度为1
np.clip(array,min,max,out)#限定值的范围在min-max之间
np.expand_dims(c,axis=(0,1))#指定位置增加为1的轴
np.dot(a,b,out=)#乘法
np.mat(a).I
np.vstack(a,b)#concatenation along the first axis
np.corrcoef
