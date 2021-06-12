from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, Dense, SimpleRNN, Flatten, LSTM
import pandas as pd
import os

class RNN:
    def __init__(self, tensor_path, prediction_offset=30, labeling_threshold=0.01):

        self.tensor_path = tensor_path
        self.tensor = self.read_tensor_from_path()

        self.features = self.tensor.shape[0]
        self.embedding_dim = 8

        self.x_train = self.get_x_train()
        self.y_train = self.get_y_train()

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
        model.add(Embedding(110, 32))
        model.add(Flatten())
        model.add(Dense(32, activation="relu"))
        model.add(Dense(1, activation='sigmoid'))

        return model

    def get_x_train(self):
        return self.tensor[self.tensor.columns[:-1]]

    def get_y_train(self):
        return self.tensor[self.tensor.columns[-1]]

    def run_model(self, save=True):
        self.model.compile(
            optimizer="rmsprop",
            loss="binary_crossentropy",
            metrics=["acc"]
        )

        self.history = self.model.fit(self.x_train, self.y_train,
                                      epochs=5, batch_size=128)

        if save:
            self.model.save_weights(os.path.join("..", "data", "model.h5"))

if __name__ == "__main__":
    rnn_model = RNN("..\\data\\tensor.joblib")
    rnn_model.run_model()
