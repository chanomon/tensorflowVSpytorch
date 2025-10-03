import tensorflow as tf
from tensorflow.keras import layers, models

# Cargar dataset MNIST
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0

# Definir modelo (API secuencial de Keras)
model = models.Sequential([
    layers.Flatten(input_shape=(28, 28)),   # aplanar imagen 28x28 -> vector
    layers.Dense(128, activation='relu'),   # capa oculta
    layers.Dropout(0.2),                   # regularización
    layers.Dense(10, activation='softmax') # salida: 10 clases
])

# Compilar y entrenar
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

model.fit(x_train, y_train, epochs=5)
print("Evaluación:", model.evaluate(x_test, y_test))

