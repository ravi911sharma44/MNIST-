import pandas as pd
import numpy as np

df = pd.read_csv (r'E:\chat bot intern\week 3\mnist_train.csv\mnist_train.csv')
test = pd.read_csv (r'E:\chat bot intern\week 3\mnist_test.csv\mnist_test.csv')

df_array = df.values
data_array = df_array[: , 1:]
target = df_array[: , 0]
target = np.array([target])
target_array = target.T
data_array = np.random.normal(0, 1e-4,data_array.shape)

print(np.linalg.matrix_rank(data_array))
print(np.linalg.matrix_rank(target_array))

A = data_array.T.dot(data_array)
A1 = np.linalg.inv(A)
B = data_array.T.dot(target_array)
C = A1.dot(B)
print(C)

test_array = test.values
test_value = test_array[: , 1:]
test_target = test_array[: , 0]
test_target = np.array([test_target])
test_target = test_target.T
test_result = test_value.dot(C)

from sklearn.metrics import mean_squared_error

print(mean_squared_error(test_target, test_result))





