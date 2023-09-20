import numpy as np
import tensorflow as tf
from tensorflow import keras

import matplotlib as mpl
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_curve, auc

from keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from keras.utils import to_categorical
from keras.models import Sequential
from keras.layers import Embedding, LSTM, Dense, Conv1D, MaxPooling1D, LSTM, Dense, Dropout, BatchNormalization
from tensorflow.keras.regularizers import l2

import os


data = pd.read_csv('TFBS.csv')
print(data.head())
print("________________________________________________________________________________________________")

train_texts, train_labels = data.iloc[:800, 0], data.iloc[:800, -1]
test_texts, test_labels = data.iloc[800:, 0], data.iloc[800:, -1]

print("train_texts and labels size: ", train_texts.shape, train_labels.shape)
print("test_texts and labels size: ", test_texts.shape, test_labels.shape)

print()
print("train_text: ", train_texts[:5])
print("text_labels: ", train_labels[:5])
print("________________________________________________________________________________________________")

# Create a tokenizer with a 10000 word vocabulary
tokenizer = Tokenizer(char_level=True)

# Fit the tokenizer on the training data
tokenizer.fit_on_texts(train_texts)

# Tokenize the training and test data
x_train = tokenizer.texts_to_sequences(train_texts)
x_test = tokenizer.texts_to_sequences(test_texts)


#print(x_test.shape)
print()
print("Tokenized X_train: ")
print(x_train[:5])
print("Tokenized X_test: ")
print(x_test[:5])

print("________________________________________________________________________________________________")


# Set the maximum sequence length
max_seq_length = max([len(seq) for seq in x_train])
print("max_seq_length",max_seq_length)


# Pad the sequences with zeros to the maximum length
x_train = pad_sequences(x_train, maxlen=max_seq_length)
x_test = pad_sequences(x_test, maxlen=max_seq_length)


y_train = train_labels
y_test = test_labels

#y_train = np.array(train_labels)
#y_test = np.array(test_labels)

print("Padded X_train: ")
print(x_train[:5])
print("Padded X_test: ")
print(x_test[:5])
print("y_train: ", y_train[:5])
print("y_test: ",y_test[:5] )
print("________________________________________________________________________________________________")

# Build the model
model = Sequential()
model.add(Embedding(500, 128, input_length=max_seq_length))
model.add(Conv1D(64, 5, activation='relu', kernel_regularizer=l2(0.01)))
model.add(MaxPooling1D(2))
model.add(LSTM(64, dropout=0.2, recurrent_dropout=0.2, return_sequences=True))
model.add(LSTM(32, dropout=0.2, recurrent_dropout=0.2))
model.add(Dense(64, activation='relu', kernel_regularizer=l2(0.01)))
model.add(Dropout(0.2))
model.add(Dense(32, activation='relu', kernel_regularizer=l2(0.01)))
model.add(Dropout(0.2))
model.add(BatchNormalization())
model.add(Dense(1, activation='sigmoid'))

model.summary()

# Compile the model
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Train the model
history = model.fit(x_train, y_train, epochs=50, batch_size=16, validation_data=(x_test, y_test), callbacks=[
        tf.keras.callbacks.EarlyStopping(monitor='val_accuracy',patience=5)
    ])

# Plotting the graph 
pd.DataFrame(history.history).plot(figsize=(8,5))
plt.grid(True)
plt.gca().set_ylim(0,1)
plt.show()

# Evaluate the model
loss, accuracy = model.evaluate(x_test, y_test)
print('Test loss:', loss)
print('Test accuracy:', accuracy)

# Make predictions on the test set
y_pred_prob = model.predict(x_test)
threshold = 0.5
y_pred = (y_pred_prob > threshold).astype(int)
print("y_pred: ",y_pred[:5])
print("y_test: ",y_test[:5])

# Compute the confusion matrix
confusion_matrix = confusion_matrix(y_test, y_pred)
print("Confusion matrix:")
print(confusion_matrix)

plt.style.use ("seaborn" )
plt.figure (figsize = (11,8) )
heatmap = sns.heatmap(confusion_matrix, linewidth = 1, annot = True, cmap="copper")
plt. show ()
print()

#print("x_test[5]: ", x_test[5])
#print("y_test: ",y_test)
