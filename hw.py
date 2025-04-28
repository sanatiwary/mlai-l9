import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

data = pd.read_csv("titanic.csv")

numCols = ["Age", "Siblings/Spouses Aboard", "Parents/Children Aboard", "Fare"]
catCols = ["Pclass", "Sex"]

for col in numCols:
    data[col].fillna(data[col].median(skipna=True), inplace=True)

for col in catCols:
    data[col].fillna(data[col].value_counts().idxmax(), inplace=True)

labelEncoder = LabelEncoder()
for col in catCols + ["Survived"]:
    data[col] = labelEncoder.fit_transform(data[col])

a = data["Survived"]
data = data.drop(["Name", "Survived"], axis=1)

scaler = MinMaxScaler()
scaler.fit(data)
scaledData = scaler.transform(data)

pca = PCA(n_components=2)
pca.fit(scaledData)
newData = pca.transform(scaledData)

xTrain, xTest, yTrain, yTest = train_test_split(newData, a, test_size=0.2, random_state=42)

model = LogisticRegression()
model.fit(xTrain, yTrain)
predictions = model.predict(xTest)

print(accuracy_score(yTest, predictions))

plt.figure(figsize=(10, 10))
plt.scatter(newData[:, 0], newData[:, 1], c=a)
plt.xlabel("first principal component")
plt.ylabel("second principal component")
plt.show()