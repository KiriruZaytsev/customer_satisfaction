import numpy as np
import pandas as pd
import random
import matplotlib.pyplot as plt

class LogRegression:

    @staticmethod
    def sigmoid(x: float) -> float:
        return 1 / (1 + np.exp(-x))
    
    @staticmethod
    def log_loss(y_true, y_pred):
        loss = 0
        for i in range(len(y_pred)):
            y = y_true[i]
            z = y_pred[i]
            loss += -y * np.log(z) - (1-y)*np.log(1-z)
        loss /= len(y_pred)
        return loss

    def __init__(self, lr = 0.01, epochs = 200):
        self.lr = lr
        self.epochs = epochs
    
    
    def fit(self, X_train, y_train):
        self.X_train = X_train
        self.y_train = y_train
        self.dim = X_train.shape[0]
        self.w = np.full((self.dim, 1), 0.01)
        self.b = 0
        self.train()

    def train(self):
        for epoch in range(self.epochs):
            dw = np.zeros((self.dim, 1))
            db = 0

            for i in range(len(self.X_train)):
                z = self.X_train[i].reshape(1, self.dim).dot(self.w) + self.b
                a = self.sigmoid(z)[0][0]

                dw += (a - self.y_train[i])*self.X_train[i].reshape(self.dim, 1)
                db += (a - self.y_train[i])

            dw /= len(self.X_train)
            db /= len(self.X_train)

            self.w -= self.lr*dw
            self.b -= self.lr*db

    def predict(self, X_test):
        self.X_test = X_test
        return np.array([self.sigmoid(x.reshape(1, self.dim).dot(self.w) + self.b)[0][0] for x in X_test])

def f(x):
    return 5*x + random.random()

def mae(y_true, predictions):
    y_true, predictions = np.array(y_true), np.array(predictions)
    return np.mean(np.abs(y_true - predictions))

def main():
    X_train = np.arange(1, 10)
    y_train = f(X_train)
    X_test = np.arange(-3, 3)
    y_test = f(X_test)
    model = LogRegression()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    print(mae(y_test, y_pred))
    plt.plot(X_test, y_test)
    plt.plot(X_test, y_test)
    plt.show()

if __name__ == "__main__":
    main()
