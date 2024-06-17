from sklearn import decomposition
import tensorflow as tf
import keras
import numpy as np

def project(signals, cell_dictionary):
    signals = signals / np.linalg.norm(signals, axis=1).reshape(-1,1)
    return(np.dot(signals, cell_dictionary))

def svd(signals, n_components=2048):
    n_components = min(n_components, signals.shape[0], signals.shape[1]) 
    pc_op = decomposition.PCA(n_components=n_components)
    data_pc = pc_op.fit_transform(signals)
    
    # normalize before autoencoder
    data_pc_std = data_pc / np.std(data_pc[:, 0])
    
    return (data_pc_std)

def run_ae(data, random_state=1234, act='relu', bias=1, dim=128, num_layers=2, dropout=0.0, lr=0.001, epochs=100, 
           val_prop=0.05, weight_decay=0, patience=10):
    
    tf.random.set_seed(random_state)
    #tf.keras.utils.set_random_seed(random_state)
    
    # encoder
    input = keras.Input(shape=(data.shape[1]))
    encoded = keras.layers.Dense(dim * 2, activation=act, use_bias=bias)(input)
    if dropout > 0:
            encoded = keras.layers.Dropout(dropout)(encoded)
    for i in range(num_layers - 2):
        encoded = keras.layers.Dense(dim * 2, activation=act, use_bias=bias)(encoded)
        if dropout > 0:
            encoded = keras.layers.Dropout(dropout)(encoded)

    encoded = keras.layers.Dense(dim, activation='linear', use_bias=bias)(encoded)

    # decoder
    decoded = keras.layers.Dense(dim * 2, activation=act,  use_bias=bias)(encoded)
    for i in range(num_layers - 2):
        decoded = keras.layers.Dense(dim * 2, activation=act,  use_bias=bias)(decoded)
    decoded = keras.layers.Dense(data.shape[1], activation='linear', use_bias=bias)(decoded)

    # autoencoder
    autoencoder = keras.Model(input, decoded)
    encoder = keras.Model(input, encoded)
    try:
        autoencoder.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=lr, decay=weight_decay), 
                        loss='mean_squared_error')
    except ValueError:
        
        autoencoder.compile(optimizer=tf.keras.optimizers.legacy.Adam(learning_rate=lr, decay=weight_decay), 
                        loss='mean_squared_error')

    callback = keras.callbacks.EarlyStopping(monitor="val_loss", patience=patience)

    history = autoencoder.fit(data, data,
                    verbose=False,
                    validation_split = val_prop,
                    epochs=epochs,
                    shuffle=True,
                    callbacks=[callback])

    embedding = encoder(data).numpy()
    
    return (embedding)
