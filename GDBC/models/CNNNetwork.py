import tensorflow as tf

class CNNNetwork:

    def custom_model(self):
        inputs = tf.keras.Input(shape=(128, 128, 1))
        x = tf.keras.layers.Conv2D(filters=32, kernel_size=16, strides=(2, 2),
                                   padding='same', activation='relu')(inputs)
        x = tf.keras.layers.MaxPool2D(pool_size=(4, 4), strides=(2, 2))(x)
        x = tf.keras.layers.Dropout(rate=0.4)(x)
        x = tf.keras.layers.Conv2D(filters=16, kernel_size=16, strides=(2, 2),
                                   padding='same', activation='relu')(x)
        x = tf.keras.layers.MaxPool2D(pool_size=(4, 4), strides=(2, 2))(x)
        x = tf.keras.layers.Dropout(rate=0.4)(x)
        x = tf.keras.layers.Flatten()(x)
        x = tf.keras.layers.Dense(units=24, activation="relu")(x)
        #x = tf.keras.layers.Dropout(rate=0.1)(x)

        outputs = tf.keras.layers.Dense(2, activation="softmax")(x)
        model = tf.keras.Model(inputs=inputs, outputs=outputs, name="my_model")
        model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.0005), loss="categorical_crossentropy",
                      metrics=["accuracy", tf.keras.metrics.Precision(), tf.keras.metrics.Recall()])
        return model

    def resnet(self):
        #resnet = tf.keras.models.load_model('/Users/charlherbst/Lemurs/lemurs/baseline/models/resnet_101V2.h5')
        resnet = tf.keras.applications.ResNet152V2(include_top=False, weights='imagenet')
        for layer in resnet.layers:
            layer.trainable = False

        inputs = tf.keras.Input(shape=(128, 128, 3))
        x = resnet(inputs)
        x = tf.keras.layers.Flatten()(x)
        #x = tf.keras.layers.Dense(units=24, activation="relu")(x)
        #x = tf.keras.layers.Dropout(rate=0.2)(x)
        outputs = tf.keras.layers.Dense(2, activation="softmax")(x)
        model = tf.keras.Model(inputs=inputs, outputs=outputs, name="my_model")
        model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.0005), loss="categorical_crossentropy",
                      metrics=["accuracy", tf.keras.metrics.AUC(), tf.keras.metrics.Precision(), tf.keras.metrics.Recall()])
        return model