import jax
from jax import random, numpy as jnp
import flax.linen as nn
import optax
import argparse
from datasets.mnist import mnist_training_dataset


_SUPPORTED_MODELS = ["lenet5", "alexnet"]


parser = argparse.ArgumentParser(
    prog="train.py",
    description="Train any classifier available in the repo.",)

parser.add_argument("model", help="Name of the model available in the repo.",
                    choices=_SUPPORTED_MODELS)
parser.add_argument("-b", "--batch_size", default=256, type=int,
                    dest="batch_size", help="Batch size to use for training.")
parser.add_argument("-e", "--epochs", default=10, type=int,
                    help="Number of epochs to train the model for.")
parser.add_argument("-lr", "--learning-rate", help="Learning rate to use.",
                    default=1e-3, type=float)

args = parser.parse_args()

if args.model == "lenet5":
    from LeNet5.model import LeNet5
    model = LeNet5()
if args.model == "alexnet":
    from AlexNet.model import AlexNet
    model = AlexNet()
else:
    raise ValueError(
        f"{args.model} is not supported. Please choose one of the following "
        f"model. {_SUPPORTED_MODELS}")

EPOCHS = args.epochs
BATCH_SIZE = args.batch_size
LEARNING_RATE = args.learning_rate

print()
print(" ======================================================")
print("|                JAX Image Classifiers                 |")
print(" ======================================================\n")
print(f"Training {model.__class__.__name__} with the following configuration:\n"
      f"Epochs: {EPOCHS} \t Batch Size: {BATCH_SIZE} \tLearning Rate {LEARNING_RATE}\n")


key = random.PRNGKey(1337)
params = model.init(key, jnp.ones((1, 28, 28, 1)))
tx = optax.adam(learning_rate=LEARNING_RATE)
opt_state = tx.init(params)


def crossentropy_loss(params, x_batched, y_batched):
    y_pred_batched = model.apply(params, x_batched)

    def loss_fn(y, y_pred):
        y = nn.one_hot(y, 10)
        return jnp.dot(y, jnp.log(y_pred))
    # vectorize
    return -jnp.mean(jax.vmap(loss_fn)(y_batched, y_pred_batched), axis=0)


loss_grad_fn = jax.value_and_grad(crossentropy_loss)


@jax.jit
def update(params, opt_state, x_batched, y_batched):
    loss, grads = loss_grad_fn(params, x_batched, y_batched)
    updates, opt_state = tx.update(grads, opt_state)
    params = jax.tree_util.tree_map(lambda p, u: p + u, params, updates)
    return params, opt_state, loss


num_batches = 0
for epoch in range(EPOCHS):
    for batch_num, (x_batch, y_batch) in enumerate(
            mnist_training_dataset(batch_size=BATCH_SIZE)):
        if epoch < 1:
            num_batches += 1
        epoch_str = f"Epoch: {str(epoch+1) + f'/' + str(EPOCHS):<8}"
        batch_str = (f"Batch: "
                     f"{str(batch_num+1) + f'/' + ('?' if epoch < 1 else str(num_batches)):<8}")
        x_batch = jnp.asarray(x_batch)
        y_batch = jnp.asarray(y_batch)
        params, opt_state, loss = update(params, opt_state, x_batch, y_batch)
        print(f"\r{epoch_str} {batch_str} Loss: {loss:<8.4f}", end="")
    print()

