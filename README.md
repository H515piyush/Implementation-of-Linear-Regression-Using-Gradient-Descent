# Implementation-of-Linear-Regression-Using-Gradient-Descent

## AIM:
To write a program to predict the profit of a city using the linear regression model with gradient descent.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1.Import the required library and read the dataframe.

2.Write a function computeCost to generate the cost function.

3.Perform iterations og gradient steps with learning rate.

4.Plot the Cost function using Gradient Descent and generate the required graph.

## Program:
/*
Program to implement the linear regression using gradient descent.
Developed by:piyush kumar
RegisterNumber:212223220075
*/
```
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

def linear_regression(X1,y,learning_rate=0.01,num_iters=1000):
    X=np.c_[np.ones(len(X1)),X1]
    theta = np.zeros(X.shape[1]).reshape(-1,1)
    for _ in range(num_iters):
        predictions = (X).dot(theta).reshape(-1,1)
        errors = (predictions-y).reshape(-1,1)
        theta-=learning_rate*(1/len(X1))*X.T.dot(errors)
    return theta
    
data=pd.read_csv('50_Startups.csv',header=None)
data.head()
X = (data.iloc[1:,:-2].values)
print(X)

X1=X.astype(float)
scaler = StandardScaler()
y=(data.iloc[1:,-1].values).reshape(-1,1)
print(y)

X1_Scaled=scaler.fit_transform(X1)
Y1_Scaled=scaler.fit_transform(y)
print(X1_Scaled)
print(Y1_Scaled)

theta = linear_regression(X1_Scaled,Y1_Scaled)

new_data=np.array([165349.2,136897.8,471784.1]).reshape(-1,1)
new_Scaled = scaler.fit_transform(new_data)
prediction = np.dot(np.append(1,new_Scaled),theta)
prediction = prediction.reshape(-1,1)
pre=scaler.inverse_transform(prediction)
print(f"Predicted Value:{pre}")
```
## Output:
![310164718-a0138386-442d-486e-9a9f-70cfc6fabbe4](https://github.com/H515piyush/Implementation-of-Linear-Regression-Using-Gradient-Descent/assets/147472999/0ad3a7b8-e161-4398-857e-0637bc4dff99)
![310167008-9533464c-d63d-4a13-8d30-78a5d6a6952d](https://github.com/H515piyush/Implementation-of-Linear-Regression-Using-Gradient-Descent/assets/147472999/ea558466-3573-4af5-a020-a1bab1db0fee)
![310167140-53a0e5e1-6413-40f4-89fd-024f0ce863ba](https://github.com/H515piyush/Implementation-of-Linear-Regression-Using-Gradient-Descent/assets/147472999/f27f477f-6d55-44e4-b581-51c24a220750)
![310167244-c2a1dd15-2a0e-4726-a6a8-c03ce9137a80](https://github.com/H515piyush/Implementation-of-Linear-Regression-Using-Gradient-Descent/assets/147472999/78cbd961-eeb4-436e-8cea-e50756f92601)
![310167352-758a524e-e2ad-457b-8b56-24158ed9017f](https://github.com/H515piyush/Implementation-of-Linear-Regression-Using-Gradient-Descent/assets/147472999/d4e41fde-1663-4a66-9521-9301dbba9160)




## Result:
Thus the program to implement the linear regression using gradient descent is written and verified using python programming.
