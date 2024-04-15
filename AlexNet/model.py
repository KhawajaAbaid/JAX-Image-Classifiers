import flax.linen as nn
import jax
import jax.numpy as jnp


class AlexNet(nn.Module):
    @nn.compact
    def __call__(self, x):
        x = nn.Conv(features=96, kernel_size=(11, 11), strides=(4, 4))(x)
        x = nn.relu(x)
        x = nn.max_pool(x, window_shape=(3, 3), strides=(2, 2))
        x = nn.Conv(features=256, kernel_size=(5, 5), padding=2)(x)
        x = nn.relu(x)
        x = nn.max_pool(x, window_shape=(3, 3), strides=(2, 2))
        x = nn.Conv(features=384, kernel_size=(3, 3), padding=1)(x)
        x = nn.relu(x)
        x = nn.Conv(features=384, kernel_size=(3, 3), padding=1)(x)
        x = nn.relu(x)
        x = nn.Conv(features=256, kernel_size=(3, 3), padding=1)(x)
        x = nn.relu(x)
        x = nn.max_pool(x, window_shape=(3, 3), strides=(2, 2))
        x = jax.vmap(jnp.ravel, in_axes=0, out_axes=0)(x)
        x = nn.Dense(4096)(x)
        x = nn.relu(x)
        x = nn.Dropout(0.5)(x)
        x = nn.Dense(4096)(x)
        x = nn.relu(x)
        x = nn.Dropout(0.5)(x)
        x = nn.Dense(10)(x)
        x = nn.softmax(x, axis=-1)
        return x
