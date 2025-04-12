import tensorflow as tf
from keras import layers, models

def create_model(input_shape):
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
    duration_output = layers.Dense(1, name="duration")(x)

    # Branch for predicting end latitude
    lat_branch = layers.Dense(128)(x)
    lat_branch = layers.BatchNormalization()(lat_branch)
    lat_branch = layers.LeakyReLU()(lat_branch)
    lat_output = layers.Dense(1, name="lat")(lat_branch)

    # Branch for predicting end longitude
    lon_branch = layers.Dense(128)(x)
    lon_branch = layers.BatchNormalization()(lon_branch)
    lon_branch = layers.LeakyReLU()(lon_branch)
    lon_output = layers.Dense(1, name="lon")(lon_branch)


    # Define the model
    model = models.Model(inputs=inputs, outputs=[duration_output, lat_output, lon_output])

    # Compile with appropriate losses and metrics
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        loss={
            "duration": "mse",
            "length": "mse",
        },
        metrics={
            "duration": ["mae"],
            "length": ["mae"],
        }
    )

    return model

def create_model_2(input_shape):
    inputs = tf.keras.Input(shape=input_shape)
    
    # First dense block with adjusted dropout rate
    x = layers.Dense(256, activation='relu')(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.35)(x)  # Adjusted dropout
    
    # Second dense block with adjusted dropout rate
    x = layers.Dense(128, activation='relu')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.25)(x)  # Adjusted dropout
    
    # Additional dense layers to boost capacity
    x = layers.Dense(64, activation='relu')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dense(32, activation='relu')(x)
    x = layers.BatchNormalization()(x)
    
    # Output branch for duration (regression)
    out_duration = layers.Dense(1, name="duration")(x)
    
    # Output branch for trip length (regression)
    out_length = layers.Dense(1, name="length")(x)
    
    model = models.Model(inputs=inputs, outputs=[out_duration, out_length])
    
    # Use loss weights to balance the two tasks if needed
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        loss={
            "duration": "mse",
            "length": "mse"
        },
        loss_weights={
            "duration": 0.5,
            "length": 0.5
        },
        metrics={
            "duration": ["mae"],
            "length": ["mae"]
        }
    )
    
    return model


def create_model_complex(input_shape):
    inputs = tf.keras.Input(shape=input_shape)

    # Shared part: Larger and deeper block with residual connections
    x = layers.Dense(512)(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.LeakyReLU()(x)
    x = layers.Dropout(0.4)(x)

    shortcut = x
    x = layers.Dense(512)(x)
    x = layers.BatchNormalization()(x)
    x = layers.LeakyReLU()(x)
    x = layers.Dropout(0.4)(x)
    x = layers.Add()([x, shortcut])

    x = layers.Dense(256)(x)
    x = layers.BatchNormalization()(x)
    x = layers.LeakyReLU()(x)
    x = layers.Dropout(0.35)(x)

    # Duration branch remains the same (using MSE)
    duration_branch = layers.Dense(128)(x)
    duration_branch = layers.BatchNormalization()(duration_branch)
    duration_branch = layers.LeakyReLU()(duration_branch)
    duration_output = layers.Dense(1, name="duration")(duration_branch)

    # Branch for predicting end latitude
    lat_branch = layers.Dense(128)(x)
    lat_branch = layers.BatchNormalization()(lat_branch)
    lat_branch = layers.LeakyReLU()(lat_branch)
    lat_output = layers.Dense(1, name="lat")(lat_branch)

    # Branch for predicting end longitude
    lon_branch = layers.Dense(128)(x)
    lon_branch = layers.BatchNormalization()(lon_branch)
    lon_branch = layers.LeakyReLU()(lon_branch)
    lon_output = layers.Dense(1, name="lon")(lon_branch)

    model = models.Model(inputs=inputs, outputs=[duration_output, lat_output, lon_output])


    model.compile(
        optimizer=tf.keras.optimizers.AdamW(learning_rate=0.001),
        loss={
            "duration": "mse",
            "lat": "mse",
            "lon": "mse"
        },
        metrics={
            "duration": ["mae"],
            "lat": ["mae"],
            "lon": ["mae"]
        },
        loss_weights={"duration": 0.33, "lat": 0.33, "lon": 0.33}  # adjust as needed
    )

    return model
