from tensorflow_addons.layers import MultiHeadAttention
import tensorflow.keras.layers as L
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import ModelCheckpoint, LearningRateScheduler, EarlyStopping
from tensorflow.keras.optimizers import Adam
import tensorflow.keras.backend as K
import gc

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
  
  outputs = L.Dense(4, activation='linear')(x)
  return Model(inputs, outputs)

def train_model(X_train, y_train, X_val, y_val, model, model_name, epochs, batch_size, save_path, save_best_only=False):
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
                                    save_best_only=save_best_only)
  
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