import pandas as pd
import numpy as np

df = pd.read_csv (r'E:\chat bot intern\week 3\mnist_train.csv\mnist_train.csv')

df_array = df.values
data_array = df_array[: , 1:]
target = df_array[: , 0]
target = np.array([target])
target_array = target.T
data_array = np.random.normal(0, 1e-4,data_array.shape)

rate = 0.01 
n_iterations = 10000
m = 60000

A = np.random.rand(784,1) 

for iteration in range(n_iterations):
    gradients = (2/m)* data_array.T.dot(data_array.dot(A) - target_array)
    A = A - rate * gradients

print(A)