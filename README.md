
## Predict Profit for food truck
![Python 2.7](https://img.shields.io/badge/Python-2.7-brightgreen.svg)    ![Problem Regression](https://img.shields.io/badge/Problem-Regression-red.svg)
#### Problem statement :
In this project, we are implementing linear regression with one
variable to predict profits for a food truck.The file ex1data1.txt contains the dataset for our linear regression problem.
The first column is the population of a city and the second column is
the profit of a food truck in that city. A negative value for profit indicates a
loss.
Dataset is like below :

| Population (10,000s) | Profit (10,000s $) |
|----------------------|--------------------|
| 5.5277               | 13.662             |
| 8.5186               | 9.1302             |
| 6.1101               | 6.8233             |

Now we have to predict profit for given population city(including which is not traverse by our food truck).

**Note:This problem statement and dataset is from coursera Andrew ng machine learning [Coursework](https://www.coursera.org/learn/machine-learning)**
#### Dependencies
* jupyter
* numpy
* matplotlib

Install dependencies using [pip](https://pip.pypa.io/en/stable/)

```python
import numpy as np
import matplotlib.pyplot as plt
```


```python
#load data from text file
data = np.loadtxt("ex1data1.txt",delimiter=",")
data = np.array(data)
```


```python
#seperate the input (X) and output (Y)
X = data[::,:1]
Y = data[::,1:]
```

#### Scatter plot  of dataset  to show how its actually looks 


```python
plt.scatter(X.transpose(),Y.transpose(),40,color="red",marker="x")
plt.xlabel("population in 10000's")
plt.ylabel("profit in 10000 $")
plt.show()
```


![png](resources/output_5_0.png)



```python
# introduce weights of hypothesis (randomly initialize)
Theta = np.random.rand(1,2)
# m is total example set , n is number of features
m,n = X.shape
# add bias to input matrix by simple make X0 = 1 for all
X_bias = np.ones((m,2))
X_bias[::,1:] = X
# output first 5 X_bias examples
X_bias[0:5,:]
```




    array([[ 1.    ,  6.1101],
           [ 1.    ,  5.5277],
           [ 1.    ,  8.5186],
           [ 1.    ,  7.0032],
           [ 1.    ,  5.8598]])




```python
#define function to find cost
def cost(X_bias,Y,Theta):
    m,n = X.shape
    hypothesis = X_bias.dot(Theta.transpose())
    return (1/(2.0*m))*((np.square(hypothesis-Y)).sum(axis=0))
```


```python
#function gradient descent algorithm from minimizing theta
def gradientDescent(X_bias,Y,Theta,iterations,alpha):
    count = 1
    cost_log = np.array([])
    while(count <= iterations):
        hypothesis = X_bias.dot(Theta.transpose())
        temp0 = Theta[0,0] - alpha*(1.0/m)*((hypothesis-Y)*(X_bias[::,0:1])).sum(axis=0)
        temp1 = Theta[0,1] - alpha*(1.0/m)*((hypothesis-Y)*(X_bias[::,-1:])).sum(axis=0)
        Theta[0,0] = temp0
        Theta[0,1] = temp1
        cost_log = np.append(cost_log,cost(X_bias,Y,Theta))
        count = count + 1
    plt.plot(np.linspace(1,iterations,iterations,endpoint=True),cost_log)
    plt.title("Iteration vs Cost graph ")
    plt.xlabel("Number of iteration")
    plt.ylabel("Cost function")
    plt.show()
    return Theta
```


```python
alpha = 0.01
iterations = 2000 #the value of iterations is 1500 enough. 2000 uses for demonstration
Theta = gradientDescent(X_bias,Y,Theta,iterations,alpha)
```


![png](resources/output_9_0.png)



```python
# predict the profit for city with 35000 and 75000 people
X_test = np.array([[1,3.5],[1,7.5]])
hypothesis = X_test.dot(Theta.transpose())
print 'profit from 35000 people city is ',hypothesis[0,0]*10000,'$'
print 'profit from 75000 people city is ',hypothesis[1,0]*10000,'$'


```

    profit from 35000 people city is  3601.80572428 $
    profit from 75000 people city is  50825.2145934 $
    


```python
# Plot showing hypothesis 
plt.scatter(X.transpose(),Y.transpose(),40,color="red",marker="x")
X_axis = X
Y_axis = X_bias.dot(Theta.transpose())
plt.plot(X_axis,Y_axis)
plt.xlabel("population in 10000's")
plt.ylabel("profit in 10000 $")
plt.show()
```


![png](resources/output_11_0.png)

