import tensorflow as tf
import numpy as np

# Define the physics-based loss function
def physics_loss(model, x, t):
    with tf.GradientTape(persistent=True) as tape:
        tape.watch(x)
        tape.watch(t)
        inputs = tf.stack([x, t], axis=1)
        preds = model(inputs)
        u = preds[:, 0]
        u_x = tape.gradient(u, x)
        u_t = tape.gradient(u, t)
        u_xx = tape.gradient(u_x, x)
        f = u_t - 0.01 * u_xx  # Heat equation with diffusion coefficient = 0.01
    return tf.reduce_mean(tf.square(f))

# Define the neural network architecture
model = tf.keras.models.Sequential([
    tf.keras.layers.Dense(50, activation="tanh", input_shape=(2,)),
    tf.keras.layers.Dense(50, activation="tanh"),
    tf.keras.layers.Dense(1)
])

# Define the training data
x = np.linspace(0, 1, 101)
t = np.linspace(0, 1, 101)
X, T = np.meshgrid(x, t)
X = X.flatten()
T = T.flatten()
u = np.sin(np.pi * X) * np.exp(-0.01 * np.pi**2 * T)

# Train the model using the physics-based loss function
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
              loss=physics_loss)
model.fit([X, T], u, epochs=10, verbose=0)
