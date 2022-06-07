import os
import pathlib
import matplotlib.pyplot as polt
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras import models

DATASET_PATH = 'data/mini_speech_commands'
data_dir = pathlib.Path(DATASET_PATH)
if not data_dir.exists():
    tf.keras.utils.get_file('mini_speech_commands.zip',
                            origin="http://storage.googleapis.com/download.tensorflow.org/data/mini_speech_commands.zip",
                            extract=True,
                            cache_dir='.', cache_subdir='data')

tf.io.gfile.listdir(str(data_dir))
data_dir
np.array(tf.io.gfile.listdir(str(data_dir)))
commands = np.array(tf.io.gfile.listdir(str(data_dir)))
commands = commands[commands != 'README.md']
commands

tf.io.gfile.glob(str(data_dir) + '/*/*')
filenames = tf.io.gfile.glob(str(data_dir) + '/*/*')
filenames = tf.random.shuffle(filenames)
filenames

len(filenames)
train_files = filenames[:6400]
val_files = filenames[6400: 6400+800]
test_files = files[-800:]
test_files = filenames[-800:]
train_files
DATASET_PATH
test_file = tf.io.read_file(train_files[10])
test_file

test_file
type(test_file)
print(test_file)
test_audio, _ = tf.audio.decode_wav(contents=test_file)

test_audio
plot(test_audio.reshape(1, -1))
test_audio.reshape(1, -1)
test_audio
test_audio.numpy()
test_audio.numpy().reshape(-1, 1)
test_audio.numpy().reshape(1, -1)
plot(test_audio.numpy().reshape(1, -1))
test_audio.numpy().reshape(1, -1)
test_audio.numpy().reshape(1, -1).shape
test_audio.numpy().shape
test_audio.numpy()[:, 0]
plot(test_audio.numpy()[:, 0])


def decode_audio(audio_binary):
    """Decode audio wav files to normalized [-1, 1] float32 arrays """
    audio, _ = tf.tf.audio.decode_wav(contents=audio_binary)
    # all the data is MONO, so drop the other channel
    return tf.squeeze(audio, axis=-1)


decode_audio(test_audio)


def decode_audio(audio_binary):
    """Decode audio wav files to normalized [-1, 1] float32 arrays """
    audio, _ = tf.audio.decode_wav(contents=audio_binary)
    # all the data is MONO, so drop the other channel
    return tf.squeeze(audio, axis=-1)


decode_audio(test_audio)
test_audio
test_files
decode_audio(test_files[30])
test_files[30]
test_files
test_files[30]
ls
decode_audio(str(test_files[30]))
test_file
decode_audio(test_file)
plot(decode_audio(test_file).numpy)
plot(decode_audio(test_file).numpy())


def get_label(file_path):
    parts = tf.strings.split(
        input=file_path,
        sep=os.path.sep)
    # Note: You'll use indexing here instead of tuple unpacking to enable this
    # to work in a TensorFlow graph.
    return parts[-2]


def get_waveform_and_label(file_path):
    label = get_label(file_path)
    audio_binary = tf.io.read_file(file_path)
    waveform = decode_audio(audio_binary)
    return waveform, label


AUTOTUNE = tf.data.AUTOTUNE
files_ds = tf.data.Dataset.from_tensor_slices(train_files)
files_ds
files_ds.shard
files_ds.shape
list(files_ds)
waveform_ds = files_ds.map(
    map_func=get_waveform_and_label, num_parallel_calls=AUTOTUNE)
waveform_ds
list(waveform_ds)
waveform_ds[0]
list(waveform_ds)[0]
list(waveform_ds)[0].numpy()
list(waveform_ds)[0][0].numpy()
plot(list(waveform_ds)[0][0].numpy())

# transform the waveforms to spectograms,


def get_spectrogram(waveform):
    # Zero-padding for an audio waveform with less than 16,000 samples.
    input_len = 16000
    waveform = waveform[:input_len]
    zero_padding = tf.zeros(
        [16000] - tf.shape(waveform),
        dtype=tf.float32)
    # Cast the waveform tensors' dtype to float32.
    waveform = tf.cast(waveform, dtype=tf.float32)
    # Concatenate the waveform with `zero_padding`, which ensures all audio
    # clips are of the same length.
    equal_length = tf.concat([waveform, zero_padding], 0)
    # Convert the waveform to a spectrogram via a STFT.
    spectrogram = tf.signal.stft(
        equal_length, frame_length=255, frame_step=128)
    # Obtain the magnitude of the STFT.
    spectrogram = tf.abs(spectrogram)
    # Add a `channels` dimension, so that the spectrogram can be used
    # as image-like input data with convolution layers (which expect
    # shape (`batch_size`, `height`, `width`, `channels`).
    spectrogram = spectrogram[..., tf.newaxis]
    return spectrogram


ls
waveform_ds.take(1)
waveform_ds.take(1)[0]
list(waveform_ds.take(1))
waveform_ds.take(1).next()
waveform_ds.take(1).next
list(waveform_ds.take(1))[0]
get_spectrogram(list(waveform_ds.take(1))[0])
get_spectrogram(list(waveform_ds.take(1))[0][0])
type(get_spectrogram(list(waveform_ds.take(1))[0][0]))
type(get_spectrogram(list(waveform_ds.take(1))[0][0]).numpy)
type(get_spectrogram(list(waveform_ds.take(1))[0][0]).numpy())
spect = get_spectrogram(list(waveform_ds.take(1))[0][0]).numpy()
spect
spect.shape
matshow(spect)
plot(spect)
plot(spect[:, :, 0])
plot(spect[:, :, 0].T)
plot(spect[:, :, 0])

spect.shape
tf.signal_stft?
tf.signal.stft?


def get_spectrogram_and_label_id(audio, label):
    spectrogram = get_spectrogram(audio)
    label_id = tf.argmax(label == commands)
    return spectrogram, label_id


spectrogram_ds = waveform_ds.map(
    map_func=get_spectrogram_and_label_id,
    num_parallel_calls=AUTOTUNE)
spectrogram_ds
len(spectrogram_ds)


def preprocess_dataset(files):
    files_ds = tf.data.Dataset.from_tensor_slices(files)
    output_ds = files_ds.map(
        map_func=get_waveform_and_label,
        num_parallel_calls=AUTOTUNE)
    output_ds = output_ds.map(
        map_func=get_spectrogram_and_label_id,
        num_parallel_calls=AUTOTUNE)
    return output_ds


# preprocess validation and test files
train_ds = spectrogram_ds
val_ds = preprocess_dataset(val_files)
test_ds = preprocess_dataset(test_files)
batch_size = 64
train_ds = train_ds.batch(batch_size)
val_ds = val_ds.batch(batch_size)
train_ds = train_ds.cache().prefetch(AUTOTUNE)
val_ds = val_ds.cache().prefetch(AUTOTUNE)
# model with simple convlolutional network
for spectrogram, _ in spectrogram_ds.take(1):
    input_shape = spectrogram.shape
num_labels = len(commands)
norm_layer = layers.Normalization()
norm_layer.adapt(data=spectrogram_ds.map(map_func=lambda spec, label: spec))
norm_layer
norm_layer.adapt_mean
norm_layer.adapt_mean.numpy()
layers.Normalization?
layers.Normalization?
model = models.Sequential([
    layers.Input(shape=input_shape),
    # downsample the input
    layers.Resizing(32, 32),
    # Normalize
    norm_layer,
    layers.Conv2D(32, 3, activation='relu'),
    layers.Conv2D(63, 3, activation='relu'),
    layers.MaxPool2D(),
    layers.Dropout(0.25),
    layers.Flatten(),
    layers.Dense(128, activation='relu'),
    layers.Dropout(0.5),
    layers.Dense(num_labels),
])
model.summary()
model.compile(optimizer=tf.keras.optimizers.Adam(),
              loss=tf.keras.losses.SparseCategoricalCrossentropy(
                  from_logits=True),
              metrics=['accuracy'],
              )
EPOCHS = 10
history = model.fit(train_ds,)
history = model.fit(train_ds,
                    validation_data=val_ds,
                    epochs=EPOCHS,
                    callbacks=tf.keras.callbacks.EarlyStopping(
                        verbose=1, patience=2),
                    )
metrics = history.history
plt.plot(history.epoch, metrics['loss'], metrics['val_loss'])
plt.legend(['loss', 'val_loss'])
paste
metrics = history.history
plt.plot(history.epoch, metrics['loss'], metrics['val_loss'])
plt.legend(['loss', 'val_loss'])
model = models.Sequential([
    layers.Input(shape=input_shape),
    # downsample the input
    layers.Resizing(128, 128),
    # Normalize
    norm_layer,
    layers.Conv2D(128, 3, activation='relu'),
    layers.Conv2D(64, 3, activation='relu'),
    layers.MaxPool2D(),
    layers.Dropout(0.25),
    layers.Flatten(),
    layers.Dense(128, activation='relu'),
    layers.Dropout(0.5),
    layers.Dense(num_labels),
])
model.compile(optimizer=tf.keras.optimizers.Adam(),
              loss=tf.keras.losses.SparseCategoricalCrossentropy(
                  from_logits=True),
              metrics=['accuracy'],
              )
history = model.fit(train_ds,
validation_data=val_ds,
epochs=EPOCHS,
callbacks=tf.keras.callbacks.EarlyStopping(verbose=1, patience=2),
)



