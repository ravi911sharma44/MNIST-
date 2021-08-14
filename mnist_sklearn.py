import pandas as pd

df = pd.read_csv (r'E:\chat bot intern\week 3\mnist_train.csv\mnist_train.csv')
df = pd.DataFrame(df)

from sklearn.model_selection import train_test_split

X = df.drop('label', axis = 1)
Y = df.label
x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size = 0.2)

from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

model = LinearRegression()
model.fit(x_train, y_train)

pred = model.predict(x_test)
print(mean_squared_error(y_test, pred))