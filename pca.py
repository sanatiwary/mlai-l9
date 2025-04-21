# pca (prinicipal component analysis) - preprocessing algorithm
# removes redundancy, inconsistency/noise, and highly correlated data
# steps: 1) standard scaling to 0-1, 2) compute covariance matrix (positive = directly proportional, negative = inversely proportional, 0 = independent), 3) finding the solution

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import load_breast_cancer

data = load_breast_cancer()
a = data["target"]
data = pd.DataFrame(data["data"], columns=data["feature_names"])

print(data.info())
print(data.head())

from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()

scaler.fit(data)
scaledData = scaler.transform(data)
print(scaledData)

from sklearn.decomposition import PCA
pca = PCA(n_components=2)

pca.fit(scaledData)
newData = pca.transform(scaledData)

print("actual data: ", scaledData.shape)
print("transformed data: ", newData.shape)

plt.figure(figsize=(10, 10))
plt.scatter(newData[:, 0], newData[:, 1], c=a)

plt.xlabel("first principal component")
plt.ylabel("second principal component")
plt.show()

