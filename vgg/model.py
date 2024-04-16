import flax.linen as nn
import jax
import jax.numpy as jnp


class VGG7(nn.Module):
    """not part of the official suite. but we include it for mnist training."""
    num_conv_layers = 4     # +3 FC layers = 7 total layers. Hence VGG7

    @nn.compact
    def __call__(self, x):
        for i in range(self.num_conv_layers):
            x = nn.Conv(features=64 * (i + 1), kernel_size=(3, 3),
                        strides=1, padding=1)(x)
            x = nn.relu(x)
            x = nn.max_pool(x, window_shape=(2, 2), strides=(2, 2))
        x = jax.vmap(jnp.ravel, in_axes=0, out_axes=0)(x)
        x = nn.Dense(4096)(x)
        x = nn.relu(x)
        x = nn.Dense(4096)(x)
        x = nn.relu(x)
        x = nn.Dense(10)(x)
        x = nn.softmax(x, axis=-1)
        return x


class VGG11(nn.Module):
    num_conv_layers = 8

    @nn.compact
    def __call__(self, x):
        for i in range(self.num_conv_layers):
            x = nn.Conv(features=64 * (i + 1), kernel_size=(3, 3),
                        strides=1, padding=1)(x)
            x = nn.relu(x)
            x = nn.max_pool(x, window_shape=(2, 2), strides=(2, 2))
        x = jax.vmap(jnp.ravel, in_axes=0, out_axes=0)(x)
        x = nn.Dense(4096)(x)
        x = nn.relu(x)
        x = nn.Dense(4096)(x)
        x = nn.relu(x)
        x = nn.Dense(10)(x)
        x = nn.softmax(x, axis=-1)
        return x


class VGG13(nn.Module):
    num_conv_layers = 10

    @nn.compact
    def __call__(self, x):
        for i in range(1, self.num_conv_layers + 1):
            x = nn.Conv(features=64 * jnp.ceil(i / 2), kernel_size=(3, 3),
                        strides=1, padding=1)(x)
            x = nn.relu(x)
            if i % 2 == 0:
                x = nn.max_pool(x, window_shape=(2, 2), strides=(2, 2))
        x = jax.vmap(jnp.ravel, in_axes=0, out_axes=0)(x)
        x = nn.Dense(4096)(x)
        x = nn.relu(x)
        x = nn.Dense(4096)(x)
        x = nn.relu(x)
        x = nn.Dense(10)(x)
        x = nn.softmax(x, axis=-1)
        return x


class VGG16(nn.Module):
    num_conv_layers = 13

    @nn.compact
    def __call__(self, x):
        features = 64
        for i in range(self.num_conv_layers):
            x = nn.Conv(features=features, kernel_size=(3, 3),
                        strides=1, padding=1)(x)
            x = nn.relu(x)
            if (i <= 4 and i % 2 == 0) or (i > 4 and i % 3 == 0):
                x = nn.max_pool(x, window_shape=(2, 2), strides=(2, 2))
                features *= 2   # Double features after every maxpool
        x = jax.vmap(jnp.ravel, in_axes=0, out_axes=0)(x)
        x = nn.Dense(4096)(x)
        x = nn.relu(x)
        x = nn.Dense(4096)(x)
        x = nn.relu(x)
        x = nn.Dense(10)(x)
        x = nn.softmax(x, axis=-1)
        return x


class VGG16A(nn.Module):
    @nn.compact
    def __call__(self, x):
        # Block 1
        x = nn.Conv(features=64, kernel_size=(3, 3), strides=1, padding=1)(x)
        x = nn.relu(x)
        x = nn.Conv(features=64, kernel_size=(3, 3), strides=1, padding=1)(x)
        x = nn.relu(x)
        x = nn.max_pool(x, window_shape=(2, 2), strides=(2, 2))
        # Block 2
        x = nn.Conv(features=128, kernel_size=(3, 3), strides=1, padding=1)(x)
        x = nn.relu(x)
        x = nn.Conv(features=128, kernel_size=(3, 3), strides=1, padding=1)(x)
        x = nn.relu(x)
        x = nn.max_pool(x, window_shape=(2, 2), strides=(2, 2))
        # Block 3
        x = nn.Conv(features=256, kernel_size=(3, 3), strides=1, padding=1)(x)
        x = nn.relu(x)
        x = nn.Conv(features=256, kernel_size=(3, 3), strides=1, padding=1)(x)
        x = nn.relu(x)
        x = nn.Conv(features=256, kernel_size=(1, 1), strides=1, padding=1)(x)
        x = nn.relu(x)
        x = nn.max_pool(x, window_shape=(2, 2), strides=(2, 2))
        # Block 4
        x = nn.Conv(features=512, kernel_size=(3, 3), strides=1, padding=1)(x)
        x = nn.relu(x)
        x = nn.Conv(features=512, kernel_size=(3, 3), strides=1, padding=1)(x)
        x = nn.relu(x)
        x = nn.Conv(features=512, kernel_size=(1, 1), strides=1, padding=1)(x)
        x = nn.relu(x)
        x = nn.max_pool(x, window_shape=(2, 2), strides=(2, 2))
        # Flatten
        x = jax.vmap(jnp.ravel, in_axes=0, out_axes=0)(x)
        # Classification head
        x = nn.Dense(4096)(x)
        x = nn.relu(x)
        x = nn.Dense(4096)(x)
        x = nn.relu(x)
        x = nn.Dense(10)(x)
        x = nn.softmax(x, axis=-1)
        return x


class VGG19(nn.Module):
    num_conv_layers = 16

    @nn.compact
    def __call__(self, x):
        features = 64
        for i in range(self.num_conv_layers):
            x = nn.Conv(features=features, kernel_size=(3, 3),
                        strides=1, padding=1)(x)
            x = nn.relu(x)
            if (i <= 4 and i % 2 == 0) or (i > 4 and i % 4 == 0):
                x = nn.max_pool(x, window_shape=(2, 2), strides=(2, 2))
                features *= 2   # Double features after every maxpool
        x = jax.vmap(jnp.ravel, in_axes=0, out_axes=0)(x)
        x = nn.Dense(4096)(x)
        x = nn.relu(x)
        x = nn.Dense(4096)(x)
        x = nn.relu(x)
        x = nn.Dense(10)(x)
        x = nn.softmax(x, axis=-1)
        return x
