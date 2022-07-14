import numpy as np
from sklearn import datasets
from sklearn.model_selection import train_test_split
from keras import models
from keras import layers
import tensorflow as tf 
import keras.backend as K
from tqdm import tqdm
import matplotlib.pyplot as plt
from utils import loop, plot_histograms

iris = datasets.load_iris()
X = iris.data
y = iris.target
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, stratify=y, random_state=42)
y_train = tf.keras.utils.to_categorical(y_train)
y_test = tf.keras.utils.to_categorical(y_test)

def build_model(act, optimizer, n_classes):
    input = tf.keras.Input(shape=(4,))
    model = tf.keras.layers.Dense(10, activation=act)(input)
    out = tf.keras.layers.Dense(n_classes, activation='softmax')(model)
    model = tf.keras.Model(inputs=input, outputs=out)
    model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])
    return model

def test_model(act, optimizer, n_classes):
    K.clear_session()
    model = build_model(act, optimizer, n_classes)
    model.fit(X_train, y_train, epochs=20, batch_size=40 , verbose=1)
    test_loss, test_acc = model.evaluate(X_test, y_test, verbose=0)
    del model
    return test_loss , test_acc

def main():
   
    test_losses_relu , test_accs_relu = loop(test_model, tf.keras.activations.relu, 1000)
    test_losses_sigmoid , test_accs_sigmoid = loop(test_model, tf.keras.activations.sigmoid, 1000)
    test_losses_tanh , test_accs_tanh = loop(test_model, tf.keras.activations.tanh, 1000)

    plot_histograms(test_losses_relu, test_accs_relu)
    plot_histograms(test_losses_sigmoid, test_accs_sigmoid)
    plot_histograms(test_losses_tanh, test_accs_tanh)

if __name__ == "__main__":
    main()