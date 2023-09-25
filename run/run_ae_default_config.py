import os, keras
import numpy as np
from keras import layers

def run_ae(data, seed=1234, act='relu', bias=1, dim=128, num_layers=2, dropout=0.0, lr=0.001, epochs=100, 
           val_prop=0.05, weight_decay=0, patience=10):
    
    keras.utils.set_random_seed(seed)
    
    # encoder
    input = keras.Input(shape=(data.shape[1]))
    encoded = layers.Dense(dim * 2, activation=act, use_bias=bias)(input)
    for i in range(num_layers - 2):
        encoded = layers.Dense(dim * 2, activation=act, use_bias=bias)(encoded)
        if dropout > 0:
            encoded = layers.Dropout(dropout)(encoded)

    encoded = layers.Dense(dim, activation='linear', use_bias=bias)(encoded)

    # decoder
    decoded = layers.Dense(dim * 2, activation=act,  use_bias=bias)(encoded)
    for i in range(num_layers - 2):
        decoded = layers.Dense(dim * 2, activation=act,  use_bias=bias)(decoded)
    decoded = layers.Dense(data.shape[1], activation='linear', use_bias=bias)(decoded)

    # autoencoder
    autoencoder = keras.Model(input, decoded)
    encoder = keras.Model(input, encoded)
    autoencoder.compile(optimizer=keras.optimizers.legacy.Adam(learning_rate=lr, decay=weight_decay), 
                        loss='mean_squared_error')

    callback = keras.callbacks.EarlyStopping(monitor="val_loss", patience=patience)

    history = autoencoder.fit(data, data,
                    validation_split = val_prop,
                    epochs=epochs,
                    shuffle=True,
                    callbacks=[callback])

    embedding = encoder(data).numpy()
    
    return (embedding)
