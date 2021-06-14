from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, Dense, SimpleRNN, Flatten, LSTM
import pandas as pd
import os


class RNN:
    def __init__(self, tensor_path, prediction_offset=30, labeling_threshold=0.01):

        self.tensor_path = tensor_path
        self.tensor = self.read_tensor_from_path()
        self.number_of_rows = self.tensor.shape[0]
        self.training_split = 0.9
        self.training_rows = round(self.training_split * self.number_of_rows)

        self.features = self.tensor.shape[0]
        self.embedding_dim = 8

        self.x_train = self.get_x_train()
        self.y_train = self.get_y_train()

        self.x_test = self.get_x_test()
        self.y_test = self.get_y_test()

        self.history = None

        self.model = self.build_model()

    @staticmethod
    def _drop_unnecessary_columns(tensor):
        col_index_to_drop = [0, 1, 3]  # unnamed index cols
        nans = tensor.isna().sum()

        for i in range(len(nans)):
            if nans[i] > 0:
                col_index_to_drop.append(i)

        return tensor.drop(tensor.columns[col_index_to_drop], axis=1)

    def read_tensor_from_path(self):
        try:
            tensor = pd.read_csv(self.tensor_path)
        except FileNotFoundError:
            raise Exception("Tensor path is wrong. Try again.")

        tensor = self._drop_unnecessary_columns(tensor)
        return tensor

    def build_model(self):

        model = Sequential()
        model.add(Dense(96, activation="relu"))
        model.add(Dense(64, activation="relu"))
        model.add(Dense(16, activation="relu"))
        model.add(Dense(1, activation='sigmoid'))

        return model

    def get_x_train(self):
        return self.tensor[self.tensor.columns[:-1]][:self.training_rows]

    def get_y_train(self):
        return self.tensor[self.tensor.columns[-1]][:self.training_rows]

    def get_x_test(self):
        return self.tensor[self.tensor.columns[:-1]][self.training_rows:]

    def get_y_test(self):
        return self.tensor[self.tensor.columns[-1]][self.training_rows:]

    def run_model(self, save=True):
        self.model.compile(
            optimizer="rmsprop",
            loss="binary_crossentropy",
            metrics=["acc"]
        )

        self.history = self.model.fit(self.x_train, self.y_train,
                                      epochs=6, batch_size=64, validation_split=0.1)

        print("Evaluate on test data")
        results = self.model.evaluate(self.x_test, self.y_test, batch_size=128)
        print("test loss, test acc:", results)

        if save:
            self.model.save_weights(os.path.join("..", "data", "model.h5"))


if __name__ == "__main__":
    rnn_model = RNN("..\\data\\tensor.joblib")
    rnn_model.run_model()
