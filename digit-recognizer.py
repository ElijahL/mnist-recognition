# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'
# %% [markdown]
# # Import Libraries

# %%
import numpy as np
import pandas as pd
import seaborn as sns 
import os
import matplotlib.pyplot as plt

np.random.seed(2)

from sklearn.model_selection import train_test_split

from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, MaxPool2D, Dropout, Dense, Flatten, BatchNormalization, LeakyReLU 
from keras.optimizers import RMSprop
from keras.utils.np_utils import to_categorical
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import ReduceLROnPlateau

sns.set(style='white', context='notebook', palette='deep')

# %% [markdown]
# # Load data

# %%
train_data = pd.read_csv('train.csv')
test_data = pd.read_csv('test.csv')


# %%
train_data.head()


# %%
train_data.shape[0]


# %%
Y_train = train_data['label']
X_train = train_data.drop(labels=['label'], axis=1)


# %%
# Free some space
del train_data


# %%
print(Y_train.value_counts())
print(sns.countplot(Y_train))

# %% [markdown]
# # Data cleaning

# %%
X_train.isnull().any().describe()


# %%
test_data.isnull().any().describe()

# %% [markdown]
# ---
# !!! NO MISSING VALUES !!!
# %% [markdown]
# # Normalize

# %%
X_train /= 255.
test_data /= 255.

# %% [markdown]
# # Reshape

# %%
img_rows, img_cols = 28, 28

X_train = X_train.values.reshape(-1, img_rows, img_cols, 1)
test_data = test_data.values.reshape(-1, img_rows, img_cols, 1)


# %%
Y_train = to_categorical(Y_train, 10)

# %% [markdown]
# # Split training and validation set

# %%
random_seed = 2


# %%
X_train, X_val, Y_train, Y_val = train_test_split(X_train, Y_train, test_size=0.1, random_state=random_seed)


# %%
plt.imshow(X_train[0][:, :, 0])

# %% [markdown]
# # Define model

# %%
model = Sequential()

# model.add(Conv2D(32, (5, 5), activation='relu', padding='same', input_shape=(img_rows, img_cols, 1)))
# model.add(BatchNormalization())
# model.add(MaxPooling2D(pool_size=(2, 2)))
# model.add(Dropout(0.25))

# model.add(Conv2D(64, (1, 1), activation='relu', padding='same'))
# model.add(BatchNormalization())
# model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
# model.add(Dropout(0.25))

# model.add(Flatten())

# model.add(Dense(512, activation='relu'))
# model.add(BatchNormalization())
# model.add(Dropout(0.5))

# model.add(Dense(10, activation='softmax'))

model.add(Conv2D(filters = 32, kernel_size = (5,5), kernel_initializer='he_normal', input_shape=(28,28,1)))
model.add(LeakyReLU(alpha = 0.2))
model.add(Conv2D(filters = 32, kernel_size = (5,5), kernel_initializer='he_normal'))
model.add(LeakyReLU(alpha = 0.2))
model.add(MaxPool2D(pool_size=(2,2), strides=(2, 2)))
model.add(Dropout(0.25))


model.add(Conv2D(filters = 64, kernel_size = (3,3), kernel_initializer='he_normal'))
model.add(LeakyReLU(alpha = 0.2))
model.add(Conv2D(filters = 64, kernel_size = (3,3), kernel_initializer='he_normal'))
model.add(LeakyReLU(alpha = 0.2))
model.add(MaxPool2D(pool_size=(1, 1), strides=(2,2)))
model.add(Dropout(0.25))


model.add(Flatten())
model.add(BatchNormalization())
model.add(Dense(512))
model.add(LeakyReLU(alpha = 0.2))
model.add(BatchNormalization())
model.add(Dropout(0.5))
model.add(Dense(10, activation = "softmax"))

model.summary()

# %% [markdown]
# # Compile

# %%
model.compile(
    loss='categorical_crossentropy',
    optimizer='adam',
    metrics=['accuracy']
)


# %%
learning_rate_reduction = ReduceLROnPlateau(
    monitor='val_accuracy',
    patience=3,
    verbose=1,
    factor=0.5,
    min_lr=1e-5
)


# %%
epochs = 25
batch_size = 86


# %%
datagen = ImageDataGenerator(
    rotation_range=10,
    zoom_range=0.1,
    width_shift_range=0.1,
    height_shift_range=0.1,
)

datagen.fit(X_train)

# %% [markdown]
# ## Fit the model

# %%
history = model.fit_generator(
    datagen.flow(X_train, Y_train, batch_size=batch_size),
    epochs=epochs,
    validation_data=(X_val, Y_val),
    verbose=2,
    steps_per_epoch=X_train.shape[0] // batch_size,
    callbacks=[learning_rate_reduction]
)


# %%
# h = model.fit(X_train, Y_train,
#           epochs=25, batch_size = None, steps_per_epoch = 37000//32,
#           validation_data = (X_val, Y_val),validation_steps = 5000//32)


# %%
model.save_weights('model.h5')


# %%
fig, ax = plt.subplots(2, 1)
ax[0].plot(history.history['loss'], color='b', label='Training loss')
ax[0].plot(history.history['val_loss'], color='r', label='Validation loss')
legend = ax[0].legend(loc='best', shadow=True)

ax[1].plot(history.history['accuracy'], color='b', label='Training accuracy')
ax[1].plot(history.history['val_accuracy'], color='r', label='Validation accuracy', axes=ax[1])
legend = ax[1].legend(shadow=True)

# %% [markdown]
# # Predict

# %%
results = model.predict(test_data)


# %%
results = np.argmax(results, axis=1)


# %%
results = pd.Series(results, name='Label')

submission = pd.concat([pd.Series(range(1,28001),name = "ImageId"),results],axis = 1)

submission.to_csv("submission.csv",index=False)


# %%
test = pd.read_csv('submission.csv')


# %%
test.head()


# %%



