import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split

data = pd.read_csv('column_2C_weka.csv')
f, ax = plt.subplots(figsize=(8, 8))
sns.heatmap(data.corr(), annot=True, linewidths=0.5, fmt=".1f", ax=ax)
plt.xticks(rotation=90)
plt.yticks(rotation=0)
plt.title('Correlation Map')
plt.show()

A = data[data["class"] == "Abnormal"]
N = data[data["class"] == "Normal"]

plt.figure(figsize=(8, 5))
plt.scatter(A.pelvic_radius, A.degree_spondylolisthesis, label="Abnormal", color="green", alpha=0.4)
plt.scatter(N.pelvic_radius, N.degree_spondylolisthesis, label="Normal", color="blue", alpha=0.4)
plt.xlabel("pelvic_radius")
plt.ylabel("degree_spondylolisthesis")
plt.legend()
plt.show()

data["class"] = [1 if each == "Abnormal" else 0 for each in data["class"]]
y = data["class"].values

x_data = data.drop(["class"], axis=1)
x = (x_data - np.min(x_data)) / (np.max(x_data) - np.min(x_data))
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=1)

score_list = []
for k in range(1, 25):
    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(x_train, y_train)
    score_list.append(knn.score(x_test, y_test))

plt.figure(figsize=(8, 5))
plt.plot(range(1, 25), score_list)
plt.xlabel("k values")
plt.ylabel("accuracy")
plt.show()
optimal_k = np.argmax(score_list) + 1
knn = KNeighborsClassifier(n_neighbors=optimal_k)
knn.fit(x_train, y_train)
prediction = knn.predict(x_test)
print("Optimal k: {}, Accuracy: {:.2f}%".format(optimal_k, knn.score(x_test, y_test) * 100))
