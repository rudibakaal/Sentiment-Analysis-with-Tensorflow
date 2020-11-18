import tensorflow as tf
from tensorflow import keras
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer
from keras.utils.vis_utils import plot_model
import matplotlib.style as style


ds = pd.read_csv('yelp_train.csv',engine='python')
ds = ds.reindex(np.random.permutation(ds.index))

train = ds

tfidf = TfidfVectorizer()
x_train = tfidf.fit_transform(train['message'].values)
x_train=x_train.toarray()

y_train = train['rating'].values

input_dim = x_train.shape[1]
model = keras.models.Sequential()
model.add(keras.layers.Dense(32, input_dim = input_dim, activation=tf.keras.layers.LeakyReLU(),kernel_initializer='he_uniform'))
model.add(keras.layers.Dense(16, activation=tf.keras.layers.LeakyReLU(),kernel_initializer='he_uniform'))
model.add(keras.layers.Dense(16, activation=tf.keras.layers.LeakyReLU(),kernel_initializer='he_uniform'))
model.add(keras.layers.Dense(1, activation='sigmoid',kernel_initializer='he_uniform'))


model.compile(optimizer='adam', loss='binary_crossentropy',
              metrics = 'binary_accuracy')


history = model.fit(x_train, y_train, epochs=35, validation_split=0.3)

metrics = np.mean(history.history['val_binary_accuracy'])
results = model.evaluate(x_train, y_train)
print('\nLoss, Binary_accuracy: \n',(results))


style.use('dark_background')
pd.DataFrame(history.history).plot(figsize=(11, 7),linewidth=4)
plt.title('Binary Cross-entropy',fontsize=14, fontweight='bold')
plt.xlabel('Epochs',fontsize=13)
plt.ylabel('Metrics',fontsize=13)
plt.show() 