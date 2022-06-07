"""https://www.tensorflow.org/text/tutorials/text_classification_rnn"""

from time import time
import matplotlib.pyplot as plt
import matplotlib as plt
import numpy as np
import tensorflow_datasets as tfds
import tensorflow as tf
tfds.disable_progress_bar()


def plot_graphs(history, metric):
    plt.plot(history.history[metric])
    plt.plot(history.history['val_'+metric], '')
    plt.xlabel("Epochs")
    plt.ylabel(metric)
    plt.legend([metric, 'val_'+metric])


dataset, info = tfds.load('imdb_reviews', with_info=True, as_supervised=True)
train_dataset, test_dataset = dataset['train'], dataset['test']

train_dataset.take(1)

for example, label in train_dataset.take(1):
    print('Text: ', example.numpy())
    print('label: ', label.numpy())

BUFFER_SIZE = 10000
BATCH_SIZE = 64

train_dataset = train_dataset.shuffle(BUFFER_SIZE).batch(
    BATCH_SIZE).prefetch(tf.data.AUTOTUNE)
test_dataset = test_dataset.batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)

VOCAB_SIZE = 1000
encoder = tf.keras.layers.TextVectorization(max_tokens=VOCAB_SIZE)
encoder.adapt(train_dataset.map(lambda text, label: text))

encoder.get_vocabulary()
encoder.get_vocabulary()[:20]
example
encoder(example)
encoder(example)[:3]
len(encoder(example))


model = tf.keras.Sequential([
    encoder,
    tf.keras.layers.Embedding(input_dim=len(encoder.get_vocabulary()),
                              output_dim=64, mask_zero=True),
    tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(64)),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dense(1)])


print([layer.supports_masking for layer in model.layers])
sample_text = ('The movie was cool. The animation and the graphics '
               'were out of this world. I would recommend this movie.')

predictions = model.predict(np.array([sample_text]))
padding = 'the' * 2000


model.compile(loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
              optimizer=tf.keras.optimizers.Adam(1e-4), metrics=['accuracy'])

history = model.fit(train_dataset, epochs=10,
                    validation_data=test_dataset, validation_steps=30)

test_loss, test_acc = model.evaluate(test_dataset)

plot_graphs(history, 'accuracy')


# second model with one bidirectional layer added
model = tf.keras.Sequential([
    encoder,
    tf.keras.layers.Embedding(input_dim=len(encoder.get_vocabulary()),
                              output_dim=64, mask_zero=True),
    tf.keras.layers.Bidirectional(
        tf.keras.layers.LSTM(64, return_sequences=True)),
    tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(32,)),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dense(1)])


model.compile(loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
              optimizer=tf.keras.optimizers.Adam(1e-4), metrics=['accuracy'])
history = model.fit(train_dataset, epochs=10,
                    validation_data=test_dataset, validation_steps=30)

plot_graphs(history, 'accuracy')

test_loss, test_acc = model.evaluate(test_dataset)

print(test_loss, test_acc)


# testing with time

t1 = time()
with tf.device('/CPU:0'):
    model = tf.keras.Sequential([
        encoder,
        tf.keras.layers.Embedding(input_dim=len(encoder.get_vocabulary()),
                                  output_dim=64, mask_zero=True),
        tf.keras.layers.Bidirectional(
            tf.keras.layers.LSTM(64, return_sequences=True)),
        tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(32,)),
        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.Dense(1)])
    model.compile(loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
                  optimizer=tf.keras.optimizers.Adam(1e-4), metrics=['accuracy'])
    history = model.fit(train_dataset, epochs=10,
                        validation_data=test_dataset, validation_steps=30)
t2 = time()
test_cpu_time = t2-t1

del model

t1 = time()
with tf.device('/GPU:0'):
    model = tf.keras.Sequential([
        encoder,
        tf.keras.layers.Embedding(input_dim=len(encoder.get_vocabulary()),
                                  output_dim=64, mask_zero=True),
        tf.keras.layers.Bidirectional(
            tf.keras.layers.LSTM(64, return_sequences=True)),
        tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(32,)),
        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.Dense(1)])
    model.compile(loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
                  optimizer=tf.keras.optimizers.Adam(1e-4), metrics=['accuracy'])
    history = model.fit(train_dataset, epochs=10,
                        validation_data=test_dataset, validation_steps=30)
t2 = time()
test_gpu_time = t2-t1

print(f'CPU time: {test_cpu_time} seconds')
print(f'GPU time: {test_gpu_time} seconds')
