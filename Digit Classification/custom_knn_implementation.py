import numpy as np
from sklearn.model_selection import train_test_split

data = np.load('./datasets/mnist_train_small.npy')

x = data[:, 1:]
y = data[:, 0]

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.33, random_state=42)



class CustomKNN:
    # constructor
    def __init__(self, n_neighbours=5):
        self.n_neighbours = n_neighbours

    # training function
    def fit(self, x, y):
        self._x = (x - x.mean()) / x.std()  # standardisation
        self._y = y

    # predict point
    # given a single point, tell me which class it belongs to
    def predict_point(self, point):
        # storing the dis of given 'point' from each point in training data
        list_dist = []

        # these points are from my training data
        for x_point, y_point in zip(self._x, self._y):
            dist_point = ((point - x_point) ** 2).sum()
            list_dist.append([dist_point, y_point])

        # sorting the list according to the distance
        sorted_dist = sorted(list_dist)
        top_k = sorted_dist[:self.n_neighbours]

        # taking the count
        items, counts = np.unique(np.array(top_k)[:, 1], return_counts=True)
        ans = items[np.argmax(counts)]
        return ans

    # predict
    # give me answer for each number in the array
    def predict(self, x):
        results = []
        x = (x - x.mean()) / x.std()
        for point in x:
            results.append(self.predict_point(point))
        return np.array(results, dtype=int)

    # score to measure my accuracy
    def score(self, x, y):
        return sum(self.predict(x) == y) / len(y)


model = CustomKNN()

model.fit(x_train, y_train)

print("accuracy is:")
print(model.score(x_test[:100], y_test[:100]) * 100, "%")
print("thankyou")






