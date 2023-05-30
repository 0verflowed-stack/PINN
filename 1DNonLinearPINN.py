import time
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from mplcursors import cursor

class PINN(tf.keras.Model):
    def __init__(self, layers):
        super(PINN, self).__init__()
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
                k_x_p_x = self.k(x, p) * p_x
            k_x_p_x_X = tape2.gradient(k_x_p_x, x)

        return p, k_x_p_x_X

    def loss_fn(self, p, k_x_p_x_X, x):
        eq = -k_x_p_x_X - self.f(x)
        bc_loss = tf.reduce_mean(tf.square(p[0])) + tf.reduce_mean(tf.square(p[-1]))
        return tf.reduce_mean(tf.square(eq)) + bc_loss

    def k(self, x, p):
        return 1 + x * p

    def f(self, x):
        return -np.ones_like(x)

# Define the network, training data, and optimizer
layers = [1, 50, 50, 50, 1] # NN architecture
model = PINN(layers)
x_train = np.linspace(0, 1, 100)[:, np.newaxis] # unflatten each value to array
optimizer = tf.keras.optimizers.Adam(learning_rate=0.005)


n_epochs = 1000
loss_array = []
best_epoch = 0
best_loss = float('inf')

start_time = time.time()
# Training loop
for epoch in range(n_epochs + 1):
    with tf.GradientTape() as tape:
        p, k_x_p_x_X = model.forward(x_train)
        loss = model.loss_fn(p, k_x_p_x_X, x_train)
    grads = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(grads, model.trainable_variables))

    if loss.numpy() < best_loss:
        best_epoch = epoch
        best_loss = loss.numpy()

    loss_array.append(loss.numpy())
    if epoch % 100 == 0:
        print(f'Epoch {epoch}, Loss: {loss.numpy()}')

print("Training took %s seconds" % (time.time() - start_time))
print(f"Best epoch: {best_epoch}, Best loss: {best_loss}")

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


# Solution from MatLab program from my coursework on 3-rd year using FEM
x_fem = np.array([0,0.05,0.1,0.15,0.2,0.25,0.3,0.35,0.4,0.45,0.5,0.55,0.6,0.65,0.7,0.75,0.8,0.85,0.9,0.95,1])
p_fem = np.array([0,-0.0239775528796244,-0.0455039546386351,-0.0646056922335109,-0.0812889350035272,-0.0955427621613265,-0.10734220544779,-0.116651344841371,-0.123426654032948,-0.127620749118809,-0.129186637552132,-0.128082485950301,-0.124276820318601,-0.117753943768429,-0.108519219278852,-0.0966037456192027,-0.0820678897925679,-0.0650031653411826,-0.0455320829887239,-0.0238058387394842,3.46944695195361e-18])

plt.figure(figsize=(8, 6))
plt.plot(x_test, p_pred, color='blue', label='PINN solution')
cursor(hover=True)
plt.xlabel('x')
plt.ylabel('p')
plt.title('Solution for the nonlinear problem')
plt.scatter(x_fem, p_fem, color='red', label='FEM solution')
plt.legend()
plt.show()