import tensorflow as tf
from tensorflow.keras import layers, models

def create_model(input_shape, num_stations):
    inputs = layers.Input(shape=input_shape)

    x = layers.Dense(64, activation="relu")(inputs)
    x = layers.Dense(32, activation="relu")(x)

    # Output 1: Trip Duration (regression)
    out_duration = layers.Dense(1, name="duration")(x)

    # Output 2: Destination Station (classification)
    out_destination = layers.Dense(num_stations, activation="softmax", name="destination")(x)

    model = models.Model(inputs=inputs, outputs=[out_duration, out_destination])

    model.compile(
        optimizer="adam",
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
