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
