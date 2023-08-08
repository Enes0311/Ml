import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

df = pd.read_csv("linear-regression-dataset.csv",sep = ";")
plt.scatter(df.experience,df.salary)
plt.xlabel("experience")
plt.ylabel("salary")
plt.show()


# sklearn library
from sklearn.linear_model import LinearRegression

# linear regression model
linear_reg = LinearRegression()

x = df.experience.values.reshape(14,1)
y = df.salary.values.reshape(14,1)

linear_reg.fit(x,y)


b0 = linear_reg.predict([[0]])
print("b0: ",b0)
b0_ = linear_reg.intercept_
print("b0_: ",b0_)   # intercept the y-axis

b1 = linear_reg.coef_
print("b1: ",b1)   # slope
new_salary = 1663 + 1138*11
print(new_salary)

b11 = linear_reg.predict([[11]])
print("b11: ",b11)
array = np.array([0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15]).reshape(-1,1)  # experience


plt.scatter(x,y)
plt.show()

y_head = linear_reg.predict(array)  # salary

plt.plot(array, y_head,color = "red")
b100 = linear_reg.predict([[100]])
print("b11: ",b100)


