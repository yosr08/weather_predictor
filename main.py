from model.model import Model


if __name__ == '__main__':
    model_instance = Model("data/weatherHistory.csv")
    model_instance.split(0.2)
    model_instance.fit()
    print(model_instance.predict())
    print("Accuracy: ", model_instance.model.score(model_instance.X_test, model_instance.y_test))