import tensorflow as tf
from vae.vae import *
import tensorflow as tf
from matplotlib import pyplot as plt
import numpy as np


checkpoint_path = '/path_to_checkpoint/cp.ckpt'

latent_dim = 32
num_units = 32
model = VAE(num_units=num_units, dim_z=latent_dim)
model.load_weights(checkpoint_path)


num_to_generate = 2000
batch_size = 500
num_batches = num_to_generate // batch_size
generated_images = []

for _ in range(num_batches):
    random_normal = np.random.normal(0, 1, size=(batch_size , 1 , latent_dim))
    x_batch = np.array(model.decoder(random_normal))
    generated_images.append(x_batch)
generated_images = np.array(generated_images).reshape([num_to_generate, 128, 128])

num_images_per_row = 10
num_rows = 10

fig, ax = plt.subplots(num_rows, num_images_per_row, figsize=(10, 10))
ax = ax.ravel()

for i in range(num_rows):
    for j in range(num_images_per_row):
        index = i * num_images_per_row + j
        if index < generated_images.shape[0]:
            ax[index].imshow(generated_images[index].reshape([128, 128]), cmap='plasma', origin='lower')
            ax[index].axis('off')

plt.subplots_adjust(wspace=0.01, hspace=0.01)
plt.show()

   

