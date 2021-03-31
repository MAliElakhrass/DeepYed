import numpy as np
from tensorflow.keras import layers, models, optimizers, callbacks
from tensorflow.python.client import device_lib


class DeepYed:
    def __init__(self):
        self.model = self.build_model(32, 4)

    def build_model(self, conv_size, conv_depth):
        board3d = layers.Input(shape=(14, 8, 8))

        # adding the convolutional layers
        x = board3d
        for _ in range(conv_depth):
            x = layers.Conv2D(filters=conv_size, kernel_size=3, padding='same', activation='relu',
                              data_format='channels_first')(x)
        x = layers.Flatten()(x)
        x = layers.Dense(64, 'relu')(x)
        x = layers.Dense(1, 'sigmoid')(x)

        return models.Model(inputs=board3d, outputs=x)

    def get_dataset(self):
        container = np.load('./data/dataset.npz')
        self.X, self.Y = container['b'], container['v']
        self.Y = np.asarray(self.Y / abs(self.Y).max() / 2 + 0.5, dtype=np.float32)  # Normalization

    def train(self):
        self.model.compile(optimizer=optimizers.Adam(5e-4), loss='mean_squared_error')
        self.model.summary()
        self.model.fit(self.X, self.Y,
                       batch_size=2048,
                       epochs=1000,
                       verbose=1,
                       validation_split=0.1,
                       callbacks=[callbacks.ReduceLROnPlateau(monitor='loss', patience=10),
                                  callbacks.EarlyStopping(monitor='loss', patience=15, min_delta=1e-4)])

        self.model.save('model.h5')


if __name__ == '__main__':
    print(device_lib.list_local_devices())
    """
    deep_yed = DeepYed()
    deep_yed.get_dataset()
    deep_yed.train()
    """
