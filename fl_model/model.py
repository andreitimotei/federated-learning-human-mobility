import tensorflow as tf
from keras import layers, models

def create_model(input_shape, num_stations):
    # Input layer
    inputs = tf.keras.Input(shape=input_shape)

    # Shared dense layers for tabular data
    x = layers.Dense(256, activation='relu')(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.4)(x)

    x = layers.Dense(128, activation='relu')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.3)(x)

    x = layers.Dense(64, activation='relu')(x)
    x = layers.BatchNormalization()(x)

    # Output branch for duration (regression)
    out_duration = layers.Dense(1, name="duration")(x)

    # Output branch for destination (classification)
    out_destination = layers.Dense(num_stations, activation="softmax", name="destination")(x)

    # Define the model
    model = models.Model(inputs=inputs, outputs=[out_duration, out_destination])

    # Compile with appropriate losses and metrics
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        loss={
            "duration": "mse",
            "destination": "sparse_categorical_crossentropy",
        },
        metrics={
            "duration": ["mae"],
            "destination": ["accuracy"],
        }
    )

    return model

def create_model_2(input_shape, num_stations):
    inputs = tf.keras.Input(shape=input_shape)
    
    # First dense block with adjusted dropout rate
    x = layers.Dense(256, activation='relu')(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.35)(x)  # Adjusted dropout from 0.4 to 0.35

    # Second dense block with adjusted dropout rate
    x = layers.Dense(128, activation='relu')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.25)(x)  # Adjusted dropout from 0.3 to 0.25

    # Additional dense layers to boost capacity (can help if underfitting)
    x = layers.Dense(64, activation='relu')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dense(32, activation='relu')(x)
    x = layers.BatchNormalization()(x)

    # Output branch for duration (regression)
    out_duration = layers.Dense(1, name="duration")(x)

    # Output branch for destination (classification)
    out_destination = layers.Dense(num_stations, activation="softmax", name="destination")(x)

    model = models.Model(inputs=inputs, outputs=[out_duration, out_destination])

    # Use loss weights to balance the two tasks (you can tune these values)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        loss={
            "duration": "mse",
            "destination": "sparse_categorical_crossentropy"
        },
        loss_weights={
            "duration": 0.6,   # Emphasize duration (regression) a bit more
            "destination": 0.4
        },
        metrics={
            "duration": ["mae"],
            "destination": ["accuracy"]
        }
    )

    return model
