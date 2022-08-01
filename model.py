import pandas as pd
from sklearn.model_selection import train_test_split
import numpy as np
from sklearn.linear_model import LinearRegression


class Model:
    def __init__(self, datafile_name):
        self.df = pd.read_csv(datafile_name)
        self.linear_reg = LinearRegression()
        self.model = None
        self.y_test = None
        self.y_train = None
        self.X_test = None
        self.X_train = None

    def split(self, test_size):
        X = np.array(self.df[['Humidity', 'Pressure (millibars)']])
        y = np.array(self.df['Temperature (C)'])
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(X, y, test_size=test_size,
                                                                                random_state=42)

    def fit(self):
        self.model = self.linear_reg.fit(self.X_train, self.y_train)

    def predict(self):
        result = self.linear_reg.predict(self.X_test)
        return result
