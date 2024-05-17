import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import time

class WaveEquationPINN(tf.keras.Model):
    def __init__(self, layers):
        super(WaveEquationPINN, self).__init__()
        self.hidden_layers = [tf.keras.layers.Dense(layer, activation=tf.nn.tanh) for layer in layers]

    def call(self, xt):
        for layer in self.hidden_layers:
            xt = layer(xt)
        return xt

@tf.function
def forward_fn(model, xt):
    xt = tf.convert_to_tensor(xt, dtype=tf.float32)
    with tf.GradientTape(persistent=True) as tape2:
        tape2.watch(xt)
        with tf.GradientTape() as tape:
            tape.watch(xt)
            u = model(xt)
        grads = tape.gradient(u, xt)
        u_x, u_t = grads[:, 0], grads[:, 1]
    grads2 = tape2.gradient(u_x, xt)
    u_xx = grads2[:, 0]
    grads3 = tape2.gradient(u_t, xt)
    u_tt = grads3[:, 1]
    del tape2
    return u, u_x, u_xx, u_t, u_tt

@tf.function
def train_step(model, xt_train, xt_ic, f, g, xt_bc, c, optimizer):
    with tf.GradientTape() as tape:
        u, u_x, u_xx, u_t, u_tt = forward_fn(model, xt_train)
        loss = loss_fn(u, u_x, u_xx, u_t, u_tt, c, model, xt_ic, f, g, xt_bc)
    grads = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(grads, model.trainable_variables))
    return loss

def loss_fn(u, u_x, u_xx, u_t, u_tt, c, model, xt_ic, f, g, xt_bc):
    wave_eq = u_tt - c ** 2 * u_xx
    ic_loss = initial_conditions(model, xt_ic, f, g)
    bc_loss = boundary_conditions(model, xt_bc)
    return tf.reduce_mean(tf.square(wave_eq)) + ic_loss + bc_loss

def initial_conditions(model, xt, f, g):
    u, _, _, u_t, _ = forward_fn(model, xt)
    u_0 = u - f(xt[:, 0])
    u_t_0 = u_t - g(xt[:, 0])
    return tf.reduce_mean(tf.square(u_0)) + tf.reduce_mean(tf.square(u_t_0))

def boundary_conditions(model, xt):
    u, _, _, _, _ = forward_fn(model, xt)
    return tf.reduce_mean(tf.square(u))

def f(x):
    return np.sin(np.pi * x)

def g(x):
    return np.zeros_like(x)

L = 1.0  # Length of the domain
c = 1.0  # Speed of wave

layers = [2, 50, 50, 100, 200, 100, 50, 50, 1]
model = WaveEquationPINN(layers)

x_train = np.linspace(0, L, 100)[:, np.newaxis]
t_train = np.linspace(0, 1, 100)[:, np.newaxis]
x_mesh, t_mesh = np.meshgrid(x_train, t_train)
xt_train = np.hstack((x_mesh.flatten()[:, np.newaxis], t_mesh.flatten()[:, np.newaxis]))

x_ic = np.linspace(0, L, 100)[:, np.newaxis]
t_ic = np.zeros_like(x_ic)
xt_ic = np.hstack((x_ic, t_ic))

x_bc_left = np.zeros_like(t_train)
x_bc_right = L * np.ones_like(t_train)
x_bc = np.concatenate((x_bc_left, x_bc_right))
t_bc = np.concatenate((t_train, t_train))
xt_bc = np.hstack((x_bc, t_bc))

batch_size = 256
train_dataset = tf.data.Dataset.from_tensor_slices(xt_train).batch(batch_size)
optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)

n_epochs = 1000
loss_array = []
start_time = time.time()

for epoch in range(n_epochs):
    for xt_batch in train_dataset:
        loss = train_step(model, xt_batch, xt_ic, f, g, xt_bc, c, optimizer)
    if epoch % 100 == 0:
        print(f'Epoch {epoch}, Loss: {loss.numpy()}')

print("Training took %s seconds" % (time.time() - start_time))

plt.plot(loss_array)
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.grid()
plt.title('Mean loss')
plt.show()

model.save_weights('pinn_wave_equation_model.h5')
