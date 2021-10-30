import glob

import pandas as pd
import numpy as np
import datetime as dt

from sklearn.preprocessing import MinMaxScaler

import tensorflow as tf
from tensorflow.keras.utils import Sequence
from tensorflow.keras.models import Model, Sequential
import tensorflow.keras.layers as L
import tensorflow.keras.backend as K
from tensorflow.keras.losses import MeanSquaredError
from tensorflow.keras.optimizers import Adam, RMSprop
from tensorflow.keras.callbacks import ModelCheckpoint, LearningRateScheduler, CSVLogger, EarlyStopping

import matplotlib.pyplot as plt
import matplotlib

import itertools
import ast
import math
import pickle
import re
import gc

from sklearn.metrics import mean_absolute_percentage_error, accuracy_score, f1_score, mean_squared_error

import itertools

from tensorflow_addons.layers import MultiHeadAttention

import logging, os
logging.disable(logging.WARNING)
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

def transformerv1_encoder(inputs, head_size, num_heads, ff_dim, dropout=0):

  x = L.LayerNormalization(epsilon=1e-6)(inputs)
  x, _ = MultiHeadAttention(head_size=head_size, num_heads=num_heads, dropout=dropout, return_attn_coef=True)([x, x])
  x = L.Dropout(dropout)(x)
  res = L.Add()([x, inputs])

  x = L.LayerNormalization(epsilon=1e-6)(res)
  x = L.Conv1D(filters=ff_dim, kernel_size=1, activation='relu')(x)
  x = L.Dropout(dropout)(x)
  x = L.Conv1D(filters=inputs.shape[-1], kernel_size=1)(x)
  x = L.Add()([x, res])

  return x


def build_transformerv1_model(input_shape, head_size, num_heads, ff_dim, 
                              num_transformer_blocks, mlp_units, dropout=0,
                              mlp_dropout=0):
  inputs = L.Input(shape=input_shape)
  x = inputs
  for _ in range(num_transformer_blocks):
    x = transformerv1_encoder(x, head_size, num_heads, ff_dim, dropout)

  x = L.GlobalAveragePooling1D(data_format="channels_first")(x)

  for dim in mlp_units:
    x = L.Dense(dim, activation='relu')(x)
    x = L.Dropout(mlp_dropout)(x)
  
  outputs = L.Dense(1, activation='linear')(x)
  return Model(inputs, outputs)

def build_bidirectional_gru(input_shape, n_units, go_backwards_list, 
middle_dense_dim=None, dropout=None, kernel_initializer='glorot_uniform'):
  model = Sequential()
  
  for i, (n_unit, go_backwards) in enumerate(zip(n_units, go_backwards_list)):
    if len(n_units) == 1:
      model.add(
          L.Bidirectional(L.GRU(n_unit, 
                      go_backwards=go_backwards,
                      kernel_initializer=kernel_initializer
                      ),
                      input_shape=input_shape,)
      )
    else:
      if i == 0:
        model.add(
            L.Bidirectional(L.GRU(n_unit, 
                        go_backwards=go_backwards, 
                        return_sequences = True,
                        kernel_initializer=kernel_initializer
                        ),
                        input_shape=input_shape,)
        )
      elif i == len(n_units) - 1:
        model.add(
            L.Bidirectional(L.GRU(n_unit, 
                        go_backwards=go_backwards, 
                        return_sequences = False,
                        kernel_initializer=kernel_initializer
                        ))
        )

      else:
        model.add(
            L.Bidirectional(L.GRU(n_unit, 
                        go_backwards=go_backwards, 
                        return_sequences = True,
                        kernel_initializer=kernel_initializer
                        ))
        )
  
  if middle_dense_dim:
    model.add(
        L.Dense(middle_dense_dim,
                activation = 'relu',
                kernel_initializer=kernel_initializer)
    )

  if dropout:
    model.add(
        L.Dropout(dropout)
    )
  
  model.add(
      L.Dense(1, activation='linear', kernel_initializer=kernel_initializer)
  )

  model.summary()

  return model

def train_model(X_train, y_train, X_val, y_val, model, model_name, epochs, batch_size, save_path):
  def lr_scheduler(epoch, lr, warmup_epochs=epochs//5, decay_epochs=epochs*2//3, initial_lr=1e-6, base_lr=1e-3, min_lr=5e-5):
    if epoch <= warmup_epochs:
        pct = epoch / warmup_epochs
        return ((base_lr - initial_lr) * pct) + initial_lr

    if epoch > warmup_epochs and epoch < warmup_epochs+decay_epochs:
        pct = 1 - ((epoch - warmup_epochs) / decay_epochs)
        return ((base_lr - min_lr) * pct) + min_lr

    return min_lr

  model.compile(optimizer=Adam(0.001), loss='mse')

  checkpoint = ModelCheckpoint(save_path + f'{model_name}.h5', 
                                    monitor='val_loss',
                                    save_best_only=False)
  
  learningrate_scheduler = LearningRateScheduler(lr_scheduler)
  early_stop = EarlyStopping(monitor='val_loss', min_delta=0, patience=30, verbose=0,)

  model.fit(X_train, y_train, 
            validation_data = (X_val, y_val), 
            batch_size = batch_size,
            epochs = epochs,
            callbacks=[
                       checkpoint,  
                       learningrate_scheduler,
                       early_stop
                       ]
            )
  
  del model
  gc.collect()
  K.clear_session()
