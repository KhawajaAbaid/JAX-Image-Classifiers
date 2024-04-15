import flax.linen as nn
import jax
from jax import numpy as jnp


class LeNet5(nn.Module):
    @nn.compact
    def __call__(self, x):
        x = nn.Conv(features=6, kernel_size=(5, 5))(x)
        x = nn.sigmoid(x)
        x = nn.avg_pool(x, window_shape=(2, 2), strides=(2, 2))
        x = nn.Conv(features=16, kernel_size=(5, 5))(x)
        x = nn.sigmoid(x)
        x = nn.avg_pool(x, window_shape=(2, 2), strides=(2, 2))
        x = jax.vmap(jnp.ravel, in_axes=0, out_axes=0)(x)
        x = nn.Dense(features=120)(x)
        x = nn.sigmoid(x)
        x = nn.Dense(features=84)(x)
        x = nn.sigmoid(x)
        x = nn.Dense(10)(x)
        return nn.softmax(x, axis=-1)
