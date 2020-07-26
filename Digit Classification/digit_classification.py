import numpy as np
from sklearn.model_selection import train_test_split

data = np.load('./datasets/mnist_train_small.npy', allow_pickle=True)

x = data[:, :1]
y = data[:, 0]

X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.33, random_state=42)

from sklearn.neighbors import KNeighborsClassifier

model = KNeighborsClassifier()

model.fit(X_train, y_train)

print("accuracy is:")
print(model.score(X_test[:100], y_test[:100]) * 100, "%")
print("thankyou")
