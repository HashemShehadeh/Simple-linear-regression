# -*- coding: utf-8 -*-
"""
Created on Sat Jul 30 21:00:22 2022

@author: hashem
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

#importing the dataset

dataset=pd.read_csv("Salary_Data.csv")

X=dataset.iloc[:,:-1].values
y=dataset.iloc[:,-1].values

from sklearn.model_selection import train_test_split

X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=0)



from sklearn.linear_model import LinearRegression

lr=LinearRegression()
lr.fit(X_train, y_train)

y_pred=lr.predict(X_test)

Hashem_salary=lr.predict(np.array(6.5).reshape(1,-1))


#Visualizing the test set results

plt.scatter(X_test, y_test,color = "red")
plt.plot(X_train, lr.predict(X_train), color = "blue")
plt.title("Salary vs Experience (test set)")
plt.xlabel("Years of Experience")
plt.ylabel("Salary")
plt.show()
