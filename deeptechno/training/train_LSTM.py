from deepTechno.model.loss import mse_with_positive_pressure_LSTM
from deepTechno.sourcing.sourcing import create_sequences
import tensorflow as tf
from deepTechno.preprocess.data_preprocess import midi_to_notes
from deepTechno.manager.manager import Manager
import pandas as pd
import numpy as np
import os

args = Manager.parse_args()

csv_file = os.path.join(os.environ.get('HOME'), args.root + args.data_dir)
# Load the CSV file into a DataFrame
df = pd.read_csv(csv_file + "/my-maestro.csv")

num_files = 128
all_notes = []

# for index, row in df.iterrows():
#     midi_file = row['midi_file']  # Assuming 'midi_file' is the name of the column containing MIDI file paths
#     try:
#         pm = pretty_midi.PrettyMIDI(midi_file)
#         # Process the PrettyMIDI object here
#         print("Successfully loaded MIDI file:", midi_file)
#     except Exception as e:
#         print("Error loading MIDI file:", midi_file)
#         print(e)

for index, row in df.iloc[:num_files].iterrows():
  print(row['midi_file'])
  notes = midi_to_notes(row['midi_file'])
  all_notes.append(notes)

all_notes = pd.concat(all_notes)

n_notes = len(all_notes)

print('Number of notes parsed:', n_notes)

key_order = ['pitch', 'step', 'duration']

train_notes = np.stack([all_notes[key] for key in key_order], axis=1)

notes_ds = tf.data.Dataset.from_tensor_slices(train_notes)
print('notes_ds', notes_ds.element_spec)

key_order = ['pitch', 'step', 'duration']

seq_length = 25
vocab_size = 128
seq_ds = create_sequences(notes_ds, seq_length, vocab_size, key_order=key_order)
print('seq_ds', seq_ds.element_spec)

for seq, target in seq_ds.take(1):
  print('sequence shape:', seq.shape)
  print('sequence elements (first 10):', seq[0: 10])
  print()
  print('target:', target)

batch_size = 64

#buffer_size = n_notes - seq_length  # the number of items in the dataset

buffer_size = 32

train_ds = (seq_ds
            .shuffle(buffer_size)
            .batch(batch_size, drop_remainder=True)
            .cache()
            .prefetch(tf.data.experimental.AUTOTUNE))

print('train_ds', train_ds.element_spec)
input_shape = (seq_length, 3)
learning_rate = 0.005

inputs = tf.keras.Input(input_shape)
x = tf.keras.layers.LSTM(128)(inputs)

outputs = {
  'pitch': tf.keras.layers.Dense(128, name='pitch')(x),
  'step': tf.keras.layers.Dense(1, name='step')(x),
  'duration': tf.keras.layers.Dense(1, name='duration')(x),
}

model = tf.keras.Model(inputs, outputs)

loss = {
      'pitch': tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
      'step': mse_with_positive_pressure_LSTM,
      'duration': mse_with_positive_pressure_LSTM,
}

optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)

model.compile(loss=loss, optimizer=optimizer)

model.summary()

losses = model.evaluate(train_ds, return_dict=True)

print('losses', losses)

model.compile(
    loss=loss,
    loss_weights={
        'pitch': 0.05,
        'step': 1.0,
        'duration':1.0,
    },
    optimizer=optimizer,
)

num_epochs = 5

# Define the checkpoint directory
checkpoint_dir = './results/training_checkpoints'

# Create a checkpoint manager
checkpoint = tf.train.Checkpoint(optimizer=model.optimizer, model=model)
checkpoint_manager = tf.train.CheckpointManager(checkpoint, checkpoint_dir, max_to_keep=5)

# Define the ModelCheckpoint callback
model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
    filepath=checkpoint_dir + '/ckpt_{epoch}.weights.h5',
    save_weights_only=True)

# Define the callbacks list
callbacks = [
    model_checkpoint_callback,
    tf.keras.callbacks.EarlyStopping(
        monitor='loss',
        patience=5,
        verbose=1,
        restore_best_weights=True),
    # Add other callbacks as needed
]

# Train the model
history = model.fit(train_ds, epochs=num_epochs, callbacks=callbacks)

# Save the final model weights using the checkpoint manager
checkpoint_manager.save()