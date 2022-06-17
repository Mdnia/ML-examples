import tensorflow as tf
from tensorflow.keras import layers
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


print(tf.__version__)
# mlp
# Class patches
# # Class patch_encoder
# create_vit_classifier
INPUT_SHAPE = (32, 32, 3)
NUM_CLASSES = 100
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar100.load_data()
print(f"X-train shape: {x_train.shape}, y_train shape: {y_train.shape}")
print(f"x_test shape: {x_test.shape}, y_test shape: {y_test.shape}")


IMAGE_SIZE = 72
PATCH_SIZE = 6
NUM_PATCHES = (IMAGE_SIZE // PATCH_SIZE) ** 2
NUM_HEADS = 4
PROJECTION_DIM = 64
TRANSFORMER_LAYERS = 8
TRANSFORMER_UNITS = [
    PROJECTION_DIM * 2,
    PROJECTION_DIM
]
BATCH_SIZE = 256
NUM_EPOCHS = 100
WEIGHT_DECAY = 0.0001
LEARNING_RATE = 0.001
MLP_HEAD_UNITS = [2048, 1024]

def mlp(input, units, dropout_rate):
    for unit in units:
        x = layers.Dense(unit, activation='gelu')(input)
        x = layers.Dropout(dropout_rate)(x)
    return x



data_augmentation = tf.keras.Sequential([
    layers.Normalization(),
    layers.Resizing(IMAGE_SIZE, IMAGE_SIZE),
    layers.RandomFlip('horizontal'),
    layers.RandomRotation(factor=0.02),
    layers.RandomZoom(
        height_factor=0.2, width_factor=0.2
    ),
], name='data_augmentation')

data_augmentation.layers[0].adapt(x_train)

class Patch_split(layers.Layer):
    def __init__(self, patch_size):
        super(Patch_split, self).__init__()
        self.patch_size = patch_size

    def __call__(self, images):

        batch_size = tf.shape(images)[0]
        patches = tf.image.extract_patches(
            images=images,
            sizes=[1, self.patch_size, self.patch_size, 1],
            strides=[1, self.patch_size, self.patch_size, 1],
            rates=[1, 1, 1, 1],
            padding='VALID'
        )
        patch_dim = patches.shape[-1]
        patches = tf.reshape(patches, [batch_size, -1, patch_dim])
        return patches

class PatchEmbedding(layers.Layer):
    def __init__(self, num_patches, projection_dim):
        super(PatchEmbedding, self).__init__()
        self.num_patches = num_patches
        self.projection = layers.Dense(units=projection_dim)
        self.positional_embedding = layers.Embedding(
            input_dim=num_patches, output_dim=projection_dim
        )

    def __call__(self, patch):
        positions = tf.range(start=0, limit=self.num_patches, delta=1)
        encoded_patches = self.projection(patch) + self.positional_embedding(positions)
        return encoded_patches


def embedded_patches():

    return

def transformer_classifier():
    inputs = layers.Input(shape=INPUT_SHAPE)
    augmented = data_augmentation(inputs)
    patches = Patch_split(PATCH_SIZE)(augmented)
    embedding = PatchEmbedding(NUM_PATCHES, PROJECTION_DIM)(patches)

    for _ in range(TRANSFORMER_LAYERS):
        norm1 = layers.LayerNormalization(epsilon=1e-6)(embedding)
        attention = layers.MultiHeadAttention(num_heads=NUM_HEADS,
                                              key_dim=PROJECTION_DIM, dropout=0.1)(norm1, norm1)
        residual1 = layers.add([embedding, attention])
        norm2 = layers.LayerNormalization(epsilon=1e-6)(residual1)
        fc = mlp(norm2, units=TRANSFORMER_UNITS, dropout_rate=0.1)
        encoded_patches = layers.add([residual1, fc])

    end_norm = layers.LayerNormalization(epsilon=1e-6)(encoded_patches)
    flatten = layers.Flatten()(end_norm)
    dropout = layers.Dropout(0.5)(flatten)

    features = mlp(dropout, units=MLP_HEAD_UNITS, dropout_rate=0.5)

    logits = layers.Dense(NUM_CLASSES)(features)

    model = tf.keras.Model(inputs=inputs, outputs=logits)

    return model


def train_model(model):



    model.compile(loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                  optimizer=tf.keras.optimizers.Adam(lr=LEARNING_RATE),
                  metrics=[
                      tf.keras.metrics.SparseCategoricalAccuracy(name='accuracy'),
                      tf.keras.metrics.SparseTopKCategoricalAccuracy(5, name='top-5-accuracy')
                  ])

    checkpoint_path = 'tmp/checkpoints'
    checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
        checkpoint_path,
        monitor='val_accuracy',
        save_best_only=True,
        save_weights_only=True
    )

    history = model.fit(x=x_train,
                        y=y_train,
                        #validation_data=(x_test, y_test),
                        epochs=NUM_EPOCHS,
                        validation_split=0.1,
                        callbacks=[checkpoint_callback])
    model.load_weights(checkpoint_path)
    _, accuracy, top_5_accuracy = model.evaluate(x_test, y_test)
    print(f'Test accuracy: {round(accuracy * 100, 2)}%')
    print(f'Top 5 test accuracy: {round(top_5_accuracy * 100, 2)}%')


    return history


if __name__ == '__main__':
    model = transformer_classifier()
    history = train_model(model)
    pd.DataFrame(history.history).plot(figsize=(10, 7))
    plt.show()