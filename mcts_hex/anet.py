import numpy as np
import tensorflow as tf
from tensorflow import keras
from keras.models import Model
from keras.layers import Input, Dense, LeakyReLU
from keras.losses import CategoricalCrossentropy



class ANET():
    """The neural network, accumulating knowledge about the 'GameWorld'."""
    def __init__(self, input_length, output_length, epochs=10, lr=0.08):
        print('ANET instance is created')
        self.lr = lr
        self.optimizer = 'sgd'
        self.epochs = epochs
        self.batch_size = 5
        self.hidden_layers = [40, 50, 30]
        self.input_length = input_length
        self.output_length = output_length
        self.model = self.generate_model()

    def generate_model(self):
        """Return a vanilla neural network according to specifications."""
        input_layer = Input(shape=(self.input_length,))
        x = Dense(self.hidden_layers[0], activation="linear")(input_layer)
        for i in range(1,len(self.hidden_layers)):
            x = Dense(self.hidden_layers[i], activation=LeakyReLU())(x)
        output_layer= Dense(self.output_length, activation="softmax")(x)
       
        model = Model(inputs=input_layer, outputs=output_layer)
        model.compile(optimizer=self.get_optimizer(), loss=CategoricalCrossentropy(), metrics=['accuracy'])
        return model

    def get_optimizer(self):
        """An overview over available optimizers."""
        if self.optimizer == 'sgd':
            return keras.optimizers.SGD(learning_rate=self.lr)
        elif self.optimizer == 'adam':
            return keras.optimizers.Adam(learning_rate=self.lr)
        elif self.optimizer == 'adagrad':
            return keras.optimizers.Adagrad(learning_rate=self.lr)
        elif self.optimizer == 'rmsprop':
            return keras.optimizers.RMSprop(learning_rate=self.lr)

    def update(self, data: np.ndarray, labels: np.ndarray) -> None:
        """Feed training data to model and train."""
        labels = self.softmax(labels) # Normalize to probability distribution
        self.model.fit(data, labels, epochs=self.epochs, batch_size=self.batch_size, verbose=0)
    
    def get_predictions(self, state: np.ndarray) -> np.ndarray:
        """Reshape state input and feed to model. Return predictions."""
        predictions: tf.Tensor = self.model(np.array([state]))[0]
        return predictions.numpy()

    def softmax(self, X: np.ndarray, theta = 1.0, axis = 1):
        """Compute the softmax of each element along an axis of X."""
        y = np.atleast_2d(X)
        if axis is None:
            axis = next(j[0] for j in enumerate(y.shape) if j[1] > 1)
        y = y * float(theta)
        y = y - np.expand_dims(np.max(y, axis = axis), axis)
        y = np.exp(y)
        ax_sum = np.expand_dims(np.sum(y, axis = axis), axis)
        p = y / ax_sum
        if len(X.shape) == 1: p = p.flatten()
        return p
 
    def save(self, id_str: str):
        self.model.save(id_str, save_format='h5')


if __name__ == '__main__':
    print('\n\n')
    anet = ANET(input_length=17, output_length=16, epochs=10, lr=0.05)