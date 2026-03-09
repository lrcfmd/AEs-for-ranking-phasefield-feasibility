import pandas as pd
import numpy as np
import tensorflow as tf
import os
from sklearn.preprocessing import StandardScaler

def run_AE(GT_vecs, Q_vecs, vector_length, k, epochs=400):
    model, history = rank(GT_vecs, vector_length=vector_length, epochs=epochs)
    save_model(model, k)   
    GT_pred = model.predict(GT_vecs)
    Q_pred = model.predict(Q_vecs)
    try:
        normalizer = tf.keras.layers.experimental.preprocessing.Normalization()
    except:
        normalizer =tf.keras.layers.Normalization()
    normalizer.adapt(GT_vecs)
    GT_vecs = normalizer(GT_vecs)
    Q_vecs = normalizer(Q_vecs)

    GT_RE = PDNB(GT_vecs, GT_pred)
    Q_RE = PDNB(Q_vecs, Q_pred)

    GT_df = pd.DataFrame(columns = ["RE"])
    Q_df = pd.DataFrame(columns = ["RE"])
    GT_df["RE"] = GT_RE
    Q_df["RE"] = Q_RE

    GT_df["RE"] = GT_df["RE"].apply(lambda x: round(x,3))

    Q_df["RE"] = Q_df["RE"].apply(lambda x: round(x,3))

    return GT_df, Q_df, history

def rank(input_x, vector_length, valids=None, epochs=400, verbose=0):
        print("Epochs = ",epochs)
        lr_fn = tf.keras.optimizers.schedules.InverseTimeDecay(
                initial_learning_rate=1e-5,
                decay_steps=100,
                decay_rate=0.9)  # increasing rate

        optim = tf.keras.optimizers.Adam(learning_rate=lr_fn)
        early = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=13)

        model = RankingAE(vector_length)
        model.normalizer.adapt(input_x)
        model.build(input_x.shape)
        model.compile(optimizer='adam')

        history = model.fit(input_x, input_x,
                  batch_size=256,
                  epochs=epochs,
                  callbacks=[early],
                  validation_split=0.05,
                  verbose=verbose
                  )

        return model, history

class RankingAE(tf.keras.Model):
    def __init__(self, vector_length):
        super(RankingAE, self).__init__()
        self.VL = vector_length
        self.normalizer = tf.keras.layers.Normalization()
        self.mse = tf.keras.losses.MeanSquaredError()
        self.encoder_layers = []
        self.dropout = tf.keras.layers.Dropout(0.1)

    def build(self, input_shape):
        seq_len = input_shape[-1]

        if self.VL == 2:
            arr = [seq_len, 1, seq_len]
        elif self.VL < 5:
            arr = [seq_len, 2, 1, 2, seq_len]
        elif self.VL < 10:
            arr = [seq_len, int(seq_len/2), int(seq_len/4), int(seq_len/2), seq_len]
        elif self.VL < 16:
            arr = [seq_len, int(seq_len/2), int(seq_len/4), 2, int(seq_len/4), int(seq_len/2), seq_len]
        elif self.VL <= 24:
            arr = [seq_len, int(seq_len/2), int(seq_len/4), 3, int(seq_len/4), int(seq_len/2), seq_len]
        elif self.VL <= 34:
            arr = [seq_len, int(seq_len/2), int(seq_len/4), int(seq_len/8), int(seq_len/4), int(seq_len/2), seq_len]
        elif self.VL <= 48:
            arr = [seq_len, int(seq_len/2), int(seq_len/4), int(seq_len/8), 4, int(seq_len/8), int(seq_len/4), int(seq_len/2), seq_len]
        elif self.VL < 72:
            arr = [seq_len, int(seq_len/2), int(seq_len/4), int(seq_len/8), int(seq_len/16), int(seq_len/8), int(seq_len/4), int(seq_len/2), seq_len]
        elif self.VL >= 72:
            arr = [seq_len, int(seq_len/2), int(seq_len/4), int(seq_len/8), int(seq_len/16), 4, int(seq_len/16), int(seq_len/8), int(seq_len/4), int(seq_len/2), seq_len]

        for n in arr:
            self.encoder_layers.append(tf.keras.layers.Dense(n, activation="relu"))
        self.output_layer = tf.keras.layers.Dense(seq_len, activation="sigmoid")

    def call(self, x):
        norm_input = self.normalizer(x)
        z = self.dropout(norm_input)
        for layer in self.encoder_layers:
            z = layer(z)
            z = self.dropout(z)
        outputs = self.output_layer(z)

        self.add_loss(self.mse(norm_input, outputs)) 

        return outputs

def PDNB(X, Y):
    euclidean_sq = np.square(Y - X)
    return np.sqrt(np.sum(euclidean_sq, axis=1)).ravel()

def save_model(m, k):
    model_save_path = os.path.join("DATA", "MODELS", str(k))
    m.save(model_save_path)
    return

def choose_existing_model(model):
    if model == None:
        df = pd.read_csv(os.path.join("DATA", "vals_df.csv"))
        k = int(df["k"].loc[df["sqrt(MFD*TPR)"].idxmax()])
    elif model == "MP":
        df = pd.read_csv(os.path.join("DATA", "vals_df.csv"))
        k = int(df["k"].iloc[-1])
    else:
        k = int(model)
    
    m = load_model(k)

    return m, k

def load_model(k_val):
    model_load_path = os.path.join("DATA", "MODELS", str(k_val))
    model = tf.keras.models.load_model(model_load_path)
    return model

def run_loaded_AE(model, Q_vecs):
    Q_pred = model.predict(Q_vecs)
    Q_vecs = model.normalizer(Q_vecs)
    re = PDNB(Q_vecs, Q_pred)
    return re 
