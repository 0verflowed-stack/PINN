import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import math
import time
from mplcursors import cursor

class PINN1D(tf.keras.Model):
    def __init__(self, layers):
        super(PINN1D, self).__init__()
        self.hidden_layers = [tf.keras.layers.Dense(layer, activation=tf.nn.tanh) for layer in layers]

    def call(self, x):
        for layer in self.hidden_layers:
            x = layer(x)
        return x

    def forward(self, x):
        x = tf.convert_to_tensor(x, dtype=tf.float32)
        with tf.GradientTape() as tape:
            tape.watch(x)
            p = self.call(x)
            with tf.GradientTape() as tape2:
                tape2.watch(x)
                with tf.GradientTape() as inner_tape:
                    inner_tape.watch(x)
                    p = self.call(x)
                p_x = inner_tape.gradient(p, x)
                k_x_p_x = k(x) * p_x
            k_x_p_x_X = tape2.gradient(k_x_p_x, x)

        return p, k_x_p_x_X

def loss_fn(p, k_x_p_x_X, f, x_bc):
    eq = -k_x_p_x_X - f(x_bc) # eq = -(k(x_bc) * p_xx) - f(x_bc)
    bc_loss = tf.reduce_mean(tf.square(p[0])) + tf.reduce_mean(tf.square(p[-1]))
    return tf.reduce_mean(tf.square(eq)) + bc_loss

# def k(x):
#     return 1 + x

# def f(x):
#     return -np.ones_like(x)

def k(x):
    eps = 1#0.015
    return 1/(4 + 1 * tf.math.sin((2*math.pi*x)/eps))

def f(x):
    return np.ones_like(x)

def exact_sol(x):
    return (-32*np.power(math.pi, 2)*np.power(x, 2) + 32*np.power(math.pi, 2)*x - 8*math.pi*x - 4*np.sin(2*math.pi*x) + (math.pi*(8*x-4)+1)*np.cos(2*math.pi*x)+4*math.pi-1)/(16*np.power(math.pi, 2))

# Define the network, training data, and optimizer
layers = [1, 50, 50, 50, 1] # NN architecture
model = PINN1D(layers)
a = 0
b = 1
x_train = np.linspace(a, b, 100)[:, np.newaxis] # unflatten each value to array
optimizer = tf.keras.optimizers.Adam(learning_rate=0.005)

# Training loop
n_epochs = 1000
loss_array = []
# add plotting values in center per epoch
middle_value_array = []
best_epoch = 0
best_loss = float('inf')

start_time = time.time()
loss = tf.constant(1)
epoch = 0
#for epoch in range(n_epochs + 1):
while loss.numpy() > 0.01:
    with tf.GradientTape() as tape:
        p, k_x_p_x_X = model.forward(x_train)
        loss = loss_fn(p, k_x_p_x_X, f, x_train)
    grads = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(grads, model.trainable_variables))

    # if loss.numpy() < best_loss:
    #     best_epoch = epoch
    #     best_loss = loss.numpy()
    #     model.save_weights('1d_pinn_linear_last_course_work_model.h5')

    loss_array.append(loss.numpy())
    if epoch % 100 == 0:
        print(f'Epoch {epoch}, Loss: {loss.numpy()}')
    epoch += 1


print("Training took %s seconds" % (time.time() - start_time))
print(f"Best epoch: {best_epoch}, Best loss: {best_loss}")

# new_model = PINN1D(layers)

# # Create some dummy input
# dummy_x = np.linspace(0, 1, 10)[:, np.newaxis]

# # Call the model on the dummy input to create variables
# _ = new_model(dummy_x)

# # Load the weights from the saved file with best loss
# loaded_model = new_model.load_weights('1d_pinn_linear_last_course_work_model.h5')

plt.plot(loss_array)
cursor(hover=True)
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.grid()
plt.title('Mean loss')
plt.show(block=False)

# Visualize the solution
x_test = np.linspace(0, 1, 100)[:, np.newaxis]
p_pred = model.predict(x_test)

x_test_exact = np.linspace(0, 1, 100)[:, np.newaxis]
p_exact = exact_sol(x_test_exact)

x_test_exact_2 = np.linspace(0, 1, 21)[:, np.newaxis]
p_exact_2 = exact_sol(x_test_exact_2)

plt.figure(figsize=(8, 6))
plt.plot(x_test, p_pred, label='PINN solution')
plt.plot(x_test, p_exact, label='Exact solution')
cursor(hover=True)
plt.xlabel('x')
plt.ylabel('p')
plt.title('Solution for the 1D linear equation')

plt.legend()
plt.show(block=False)

plt.figure(figsize=(8, 6))
plt.plot(x_test, p_pred, label='PINN solution')
plt.scatter(x_test_exact_2, p_exact_2, color='red', label='Exact solution', s=10)
cursor(hover=True)
plt.xlabel('x')
plt.ylabel('p')
plt.title('Solution for the 1D linear equation')

plt.legend()
plt.show()
