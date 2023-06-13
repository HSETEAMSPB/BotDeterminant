from config import *
from make_dataset import *
from model import ASR
from optimizer import make_opt
from loss import rnn_transducer_loss

import tensorflow as tf


def dev_step(x, y, x_len, y_len):
    logits, x_len, y_len = model(x, y, x_len, y_len, training=False)
    if not tf.config.list_physical_devices("GPU"):
        logits = tf.nn.log_softmax(logits)
    loss = rnn_transducer_loss(logits, y, y_len, x_len)
    loss = loss / tf.cast(y_len, dtype=tf.float32)
    
    return tf.reduce_mean(loss)


def train_step(x, y, x_len, y_len):
    with tf.GradientTape() as tape:
        logits, x_len, y_len = model(x, y, x_len, y_len, training=True)
        if not tf.config.list_physical_devices("GPU"):
            logits = tf.nn.log_softmax(logits)

        y = tf.cast(y, dtype=tf.int64)
        x_len = tf.cast(x_len, dtype=tf.int64)
        y_len = tf.cast(y_len, dtype=tf.int64)
        loss = rnn_transducer_loss(logits, y, y_len, x_len)

    variables = model.trainable_variables
    grads = tape.gradient(loss, model.trainable_variables)

    optimizer.apply_gradients(zip(grads, variables))
    return tf.math.reduce_mean(loss)


def train():
    global model, optimizer
    model = ASR(
        num_units=num_units,
        num_vocabulary=num_vocab,
        count_lstms=num_lstms,
        lstm_units=lstm_units,
        output_dim=out_dim,
    )

    if not os.path.exists(f"{librispeech_dir}/dev"):
        make_vocabulary()
        download_and_split()

    dev_dataset = dataset("dev")
    train_dataset = dataset("train")
    dev_size, train_size = len(list(dev_dataset)), len(list(train_dataset))
    step = tf.Variable(1)
    optimizer = make_opt()
    checkpoint = tf.train.Checkpoint(step=step, optimizer=optimizer, model=model)
    ckpt_manager = tf.train.CheckpointManager(checkpoint, "./ckpt", max_to_keep=10)

    for epoch in range(epochs):
        train_batches = 0
        for x, y, x_len, y_len in train_dataset:
            loss = train_step(x, y, x_len, y_len)
            train_batches += 1

            if step % 1000 == 0:
                dev_batches = 0
                for x, y, x_len, y_len in dev_dataset:
                    loss = dev_step(x, y, x_len, y_len)
                    dev_batches += 1
                    print(
                        f"[dev]   epoch: {epoch+1:3d} batch: {dev_batches:5d}/{dev_size} loss: {loss}"
                    )
                ckpt_manager.save()
                print("[checkpoint has saved]")

            step.assign_add(1)
            print(
                f"[train] epoch: {epoch+1:3d} batch: {train_batches:5d}/{train_size} loss: {loss}"
            )


if __name__ == "__main__":
    train()
