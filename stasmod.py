import statsmodels.api as sm
import  statsmodels.formula.api as smf
import  pandas as pd
import  numpy as np
import matplotlib.pyplot as plt

data=pd.DataFrame({"X":np.arange(10,20,0.25)})
data["Y"]=2*data["X"]+1+np.random.randn(40)
print(data.head())
mod=smf.ols("Y~X",data).fit()
print(mod.summary())
data.plot(x="X",y="Y",kind="scatter",figsize=(8,5))
plt.plot(data["X"],mod.params[0]+mod.params[1]*data["X"],"r")
plt.show()


