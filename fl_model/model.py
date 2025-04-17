import tensorflow as tf
from keras import layers, models
import tensorflow as tf

import pickle

with open("data/geohash_le.pkl", "rb") as f:
    geohash_le = pickle.load(f)
n_geohash_buckets = len(geohash_le.classes_)

with open("data/dest_le.pkl", "rb") as f:
    dest_le = pickle.load(f)
n_top_dest = len(dest_le.classes_)


def geodesic_loss(y_true, y_pred):
    """
    Compute the mean geodesic (haversine) distance between predicted and true coordinates.
    Both y_true and y_pred are tensors of shape (batch_size, 2) where the first column is latitude 
    and the second column is longitude, expressed in degrees.
    """
    # Conversion factor for degrees to radians
    deg_to_rad = tf.constant(3.141592653589793 / 180, dtype=tf.float32)

    # Convert degrees to radians
    lat_true = y_true[:, 0] * deg_to_rad
    lon_true = y_true[:, 1] * deg_to_rad
    lat_pred = y_pred[:, 0] * deg_to_rad
    lon_pred = y_pred[:, 1] * deg_to_rad

    # Compute differences
    dlat = lat_pred - lat_true
    dlon = lon_pred - lon_true

    # Haversine formula
    a = tf.sin(dlat / 2)**2 + tf.cos(lat_true) * tf.cos(lat_pred) * tf.sin(dlon / 2)**2
    c = 2 * tf.atan2(tf.sqrt(a), tf.sqrt(1 - a))
    R = 6378137.0  # Earth's radius in meters
    distance = R * c  # in meters

    return tf.reduce_mean(distance)


def create_model_complex(input_shape):
    inputs = tf.keras.Input(shape=input_shape)
    
    # Shared part: Larger and deeper block with residual connections
    x = layers.Dense(1024, kernel_regularizer=tf.keras.regularizers.l2(0.01))(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('gelu')(x)
    x = layers.Dropout(0.5)(x)

    shortcut = x
    x = layers.Dense(1024, kernel_regularizer=tf.keras.regularizers.l2(0.01))(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('gelu')(x)
    x = layers.Dropout(0.5)(x)
    x = layers.Add()([x, shortcut])

    x = layers.Dense(512, activation='relu')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.4)(x)

    x = layers.Dense(256, activation='relu')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.3)(x)
    
    # Instead of separate branches, create a single output that predicts both coordinates.
    outputs = layers.Dense(2, name="coords")(x)
    
    model = models.Model(inputs=inputs, outputs=outputs)

    # Use an exponential decay learning rate scheduler for AdamW
    lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
        initial_learning_rate=0.001,
        decay_steps=10000,
        decay_rate=0.9
    )
    
    model.compile(
        optimizer=tf.keras.optimizers.AdamW(learning_rate=lr_schedule),
        loss=geodesic_loss,
        metrics=[geodesic_loss]  # You can add more metrics if desired.
    )
    
    return model


def create_transformer_model(input_shape):
    # Input layer
    inputs = tf.keras.Input(shape=input_shape)

    # Split out the first 5 numeric vs last 4 categorical features
    numeric_features = inputs[:, 0:5]  # Start_date, Start_dayofweek, Start_lat, Start_lon, Bike number
    categorical_geohash = inputs[:, 5]
    categorical_topdest = inputs[:, 8]
    additional_features = inputs[:, 6:8]  # station_avg_dur, station_std_dur

    # Use Lambda layers to cast categorical features to int32
    categorical_geohash = layers.Lambda(lambda x: tf.cast(x, tf.int32))(categorical_geohash)
    categorical_topdest = layers.Lambda(lambda x: tf.cast(x, tf.int32))(categorical_topdest)

    # Embed the geohash and top destination
    geohash_embed = layers.Embedding(input_dim=geohash_le.classes_.size, output_dim=8)(categorical_geohash)
    topdest_embed = layers.Embedding(input_dim=dest_le.classes_.size, output_dim=8)(categorical_topdest)

    # Concatenate all features
    x = layers.Concatenate()([numeric_features, additional_features, geohash_embed, topdest_embed])

    # Shared part: Larger and deeper block with residual connections
    x = layers.Dense(1024, kernel_regularizer=tf.keras.regularizers.l2(0.01))(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('gelu')(x)
    x = layers.Dropout(0.5)(x)

    shortcut = x
    x = layers.Dense(1024, kernel_regularizer=tf.keras.regularizers.l2(0.01))(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('gelu')(x)
    x = layers.Dropout(0.5)(x)
    x = layers.Add()([x, shortcut])

    x = layers.Dense(512, activation='relu')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.4)(x)

    x = layers.Dense(256, activation='relu')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.3)(x)

    # Final output layer: predicts two values (latitude and longitude)
    outputs = layers.Dense(2, name="coords")(x)

    # Compile the model
    model = models.Model(inputs=inputs, outputs=outputs)
    model.compile(
        optimizer=tf.keras.optimizers.AdamW(learning_rate=0.001),
        loss=geodesic_loss,
        metrics=[geodesic_loss]
    )

    return model
