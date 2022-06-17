import tensorflow as tf
from tensorflow.keras import layers
import matplotlib.pyplot as plt
import numpy as np

print(tf.__version__)
# mlp
# Class patches
# # Class patch_encoder
# create_vit_classifier

(x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar100.load_data()
print(f"X-train shape: {x_train.shape}, y_train shape: {y_train.shape}")
print(f"x_test shape: {x_test.shape}, y_test shape: {y_test.shape}")


IMAGE_SIZE = 72
PATCH_SIZE = 6
NUM_PATCHES = (IMAGE_SIZE // PATCH_SIZE) ** 2
NUM_HEADS = 4


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


class Patch_split(layers.Layer):
    def __init__(self, patch_size):
        super(Patch_split, self).__init__()
        self.patch_size = patch_size

    def call(self, images):

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

    def call(self, patch):
        positions =tf.range(start=0, limit=self.num_patches, delta=1)
        encoded_patches = self.projection(patch) + self.positional_embedding(positions)
        return encoded_patches


def embedded_patches():


def transformer_classifier():



if __name__ == '__main__':
    plt.figure(figsize=(4, 4))
    image = x_train[np.random.choice(range(x_train.shape[0]))]
    plt.imshow(image.astype('uint8'))
    plt.axis('off')
    plt.show()