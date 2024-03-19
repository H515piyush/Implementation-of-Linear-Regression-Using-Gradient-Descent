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
```/*
Program to implement the linear regression using gradient descent.
Developed by:piyush kumar
RegisterNumber:212223220075
*/
```
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
profile prediction
![308782105-5d6e3ac5-f8b1-4e73-b433-6697f51f71bc](https://github.com/H515piyush/Implementation-of-Linear-Regression-Using-Gradient-Descent/assets/147472999/e680620f-62ff-4828-9b30-37ae5ba08215)
FUNCTION:
![308782233-0faf5afd-fa49-4d20-b7e5-5ad223278a9a](https://github.com/H515piyush/Implementation-of-Linear-Regression-Using-Gradient-Descent/assets/147472999/cf004022-3150-4f7e-b724-3502f3d6283f)
Gradient descent:
![308783411-74811aed-d411-423c-8d1f-5b5e4ca2cbb9](https://github.com/H515piyush/Implementation-of-Linear-Regression-Using-Gradient-Descent/assets/147472999/dfb99c54-a8e9-4693-b304-665473e07c15)
cost function using gradient descent:
![308782429-e6873033-1c67-4b0a-87d7-ec6d6643e61b](https://github.com/H515piyush/Implementation-of-Linear-Regression-Using-Gradient-Descent/assets/147472999/4bb8d161-8c9c-46ef-84cf-f23f33cc336b)
linear regression using profile prediction:
![308782535-95675739-0f42-4f8c-b660-8ee7c40afef4](https://github.com/H515piyush/Implementation-of-Linear-Regression-Using-Gradient-Descent/assets/147472999/b7cfee9f-4a33-4617-8c96-27906203caa9)
profile presiction for the population of 35000:
![308782655-ac1e96c6-b4e3-4216-8a60-877e6f9ff5d0](https://github.com/H515piyush/Implementation-of-Linear-Regression-Using-Gradient-Descent/assets/147472999/1ac5906b-fc55-4885-9e3c-113bb94ae387)##profile prediction for the population of 70000:


![308782822-2d0918f6-11d2-441f-80f9-04baa1772352](https://github.com/H515piyush/Implementation-of-Linear-Regression-Using-Gradient-Descent/assets/147472999/3f3053c6-bef3-4e4a-bc8f-849c211ba851)
## Result:
Thus the program to implement the linear regression using gradient descent is written and verified using python programming.
