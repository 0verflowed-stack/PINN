import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import time
import math
from matplotlib.patches import Rectangle
from matplotlib import cm

class Puasson2DPINN(tf.keras.Model):
    def __init__(self, layers):
        super(Puasson2DPINN, self).__init__()
        self.hidden_layers = [tf.keras.layers.Dense(layer, activation=tf.nn.tanh) for layer in layers]

    def call(self, x1_x2):
        for layer in self.hidden_layers:
            x1_x2 = layer(x1_x2)
        return x1_x2

    def forward(self, x1_x2):
        x1_x2 = tf.convert_to_tensor(x1_x2, dtype=tf.float32)
        
        with tf.GradientTape(persistent=True) as tape2:
            tape2.watch(x1_x2)
            with tf.GradientTape() as tape:
                tape.watch(x1_x2)
                u = self.call(x1_x2)
            grads = tape.gradient(u, x1_x2)
            u_x1, u_x2 = grads[:, 0], grads[:, 1]
        grads2 = tape2.gradient(u_x1, x1_x2)
        u_x1_x1 = grads2[:, 0]

        grads3 = tape2.gradient(u_x2, x1_x2)
        u_x2_x2 = grads3[:, 1]

        del tape2

        return u, u_x1, u_x2, u_x1_x1, u_x2_x2

    def dirichlet_condition(self, model, x1_x2_bc_up_down):
        u, _, _, _, _ = model.forward(x1_x2_bc_up_down)
        return tf.reduce_mean(tf.square(u))

    def neuman_condition(self ,model, x1_x2_bc_left_right):
        _, u_x1, _,  _, _ = model.forward(x1_x2_bc_left_right)
        return tf.reduce_mean(tf.square(u_x1)) # -∂u/∂x1 + ∂u/∂x1


    def loss_fn(self, u_x1_x1, u_x2_x2, model, f, x1_x2_bc_left_right, x1_x2_bc_up_down):
        puasson_eq = u_x1_x1 + u_x2_x2 - f(x1_x2_train)
        bc_dirichlet = self.dirichlet_condition(model, x1_x2_bc_up_down)
        bc_neuman = self.neuman_condition(model, x1_x2_bc_left_right)

        return 1 * tf.reduce_mean(tf.square(puasson_eq)) + 2 * (bc_dirichlet + bc_neuman)

# Right-hand side of Puasson equation
def f_2D(x1_x2):
    x1 = np.array([x[0] for x in x1_x2])
    x2 = np.array([x[1] for x in x1_x2])
    return -np.pi * np.sin(np.pi * x1) * np.sin(np.pi * x2)

# Function to compute the exact solution
def exact_solution_2D(x1, x2):
    return np.sin(np.pi * x1) * np.sin(np.pi * x2)#np.power(np.pi, 2) * np.sin(np.pi * x2) #(x1*(1-x1))/2

print("Available devices:", tf.config.list_physical_devices())

L_x1 = 1.0 # right side of x
L_x2 = 1.0

# Define the network, training data, and optimizer
layers = [10, 10, 10, 1]
model = Puasson2DPINN(layers)
x1_train = np.linspace(0, L_x1, 20)[:, np.newaxis]
x2_train = np.linspace(0, L_x2, 20)[:, np.newaxis]
x1_mesh, x2_mesh = np.meshgrid(x1_train, x2_train)
x1_x2_train = np.hstack((x1_mesh.flatten()[:, np.newaxis], x2_mesh.flatten()[:, np.newaxis]))


optimizer = tf.keras.optimizers.Adam(learning_rate=0.005)

# bc - boundary condition (left and right) neuman
#         _______________________
#         ||                       ||
# [-1, 0] || x2                    || [1, 0] 
#         ||          x1           ||
#         ||_______________________||
# ∂u/∂n = -∂u/∂x1                      ∂u/∂n = ∂u/∂x1
x1_bc_left = np.zeros_like(x2_train)
x1_bc_right = L_x1 * np.ones_like(x2_train)
x1_bc_left_right = np.concatenate((x1_bc_left, x1_bc_right))
x2_bc_left_right = np.concatenate((x2_train, x2_train))
x1_x2_bc_left_right = np.hstack((x1_bc_left_right, x2_bc_left_right))

# bc - boundary condition (up and down) dirichlet
#          [0, 1]  ∂u/∂n = ∂u/∂x2
#  =======================
# |                       |
# | x2                    |
# |          x1           |
# |=======================|
#          [0, -1]  ∂u/∂n = -∂u/∂x2
x2_bc_up = np.zeros_like(x1_train)
x2_bc_down = L_x1 * np.ones_like(x1_train)
x2_bc_up_down = np.concatenate((x2_bc_up, x2_bc_down))
x1_bc_up_down = np.concatenate((x1_train, x1_train))
x1_x2_bc_up_down = np.hstack((x1_bc_up_down, x2_bc_up_down))

# Training loop
n_epochs = 300
loss_array = []
start_time = time.time()

loss = tf.constant(1)
epoch = 0
#for epoch in range(n_epochs + 1):
while loss.numpy() > 0.001:
    with tf.GradientTape() as tape:
        u, u_x1, u_x2, u_x1_x1, u_x2_x2 = model.forward(x1_x2_train)
        loss = model.loss_fn(u_x1_x1, u_x2_x2, model, f_2D, x1_x2_bc_left_right, x1_x2_bc_up_down)
    grads = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(grads, model.trainable_variables))

    loss_array.append(loss)
    if epoch % 100 == 0:
        print(f'Epoch {epoch}, Loss: {loss.numpy()}')
    epoch += 1

print("Training took %s seconds" % (time.time() - start_time))
print(f"Last epoch: {epoch}, loss: {loss.numpy()}")

plt.plot(loss_array)
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.grid()
plt.title('Mean loss')
plt.show(block=False)

# Compute the exact solution on the test grid
U_exact = exact_solution_2D(x1_mesh, x2_mesh)

# 3D plot of the exact solution
fig_exact = plt.figure()
ax_exact = fig_exact.add_subplot(111, projection='3d')
ax_exact.plot_surface(x1_mesh, x2_mesh, U_exact, cmap='viridis')
ax_exact.set_xlabel('x_1')
ax_exact.set_ylabel('x_2')
ax_exact.set_zlabel('u')
ax_exact.set_title('Exact Solution for the Wave Equation')
plt.show(block=False)

# Save weights to file
model.save_weights('pinn_wave_equation_model.h5')

# Instantiate a new model with the same layers and wave speed
new_model = Puasson2DPINN(layers)

# Create some dummy input
dummy_x = np.linspace(0, 1, 10)
dummy_t = np.linspace(0, 1, 10)
dummy_xt = np.array([[x1, x2] for x1 in dummy_x for x2 in dummy_t], dtype=np.float32)

# Call the model on the dummy input to create variables
_ = new_model(dummy_xt)

# Load the weights from the saved file
loaded_model = new_model.load_weights('pinn_wave_equation_model.h5')

# Visualize the solution
x_test = np.linspace(0, L_x1, 100)[:, np.newaxis]
x2_test = np.linspace(0, L_x2, 100)[:, np.newaxis]
x1_mesh, x2_mesh = np.meshgrid(x_test, x2_test)
x1x2_test = np.hstack((x1_mesh.flatten()[:, np.newaxis], x2_mesh.flatten()[:, np.newaxis]))

u_pred = new_model.predict(x1x2_test).reshape(x_test.shape[0], x2_test.shape[0])

# Compute the exact solution for the test points
u_exact = exact_solution_2D(x1_mesh, x2_mesh)

# Compute the mean square error between the predicted solution and the exact solution 
mse = np.mean((u_pred - u_exact)**2)
print("Mean Square Error: ", mse)

plt.figure(figsize=(8, 6))
plt.pcolor(x1_mesh, x2_mesh, u_pred, cmap='viridis')
plt.colorbar()
plt.xlabel('x_1')
plt.ylabel('x_2')
plt.title('PINN Solution for the Puasson Equation')
plt.show(block=False)


fig = plt.figure(figsize=(10, 6))
ax = fig.add_subplot(111, projection='3d')

# line, = ax.plot(np.ones_like(x_test_1D) * 0.5, x_test_1D, u_pred_1D, label='PINN Solution 1D', zorder=2)
# line2, = ax.plot(np.ones_like(x_test_1D) * 0.5, x_test_1D, U_exact_1D, label='Exact Solution 1D', zorder=1)
surf = ax.plot_surface(x1_mesh, x2_mesh, u_pred, cmap='viridis', linewidth=0, antialiased=False, label='PINN Solution 2D', zorder=0)
#ax.plot(np.ones_like(x_test_1D) * 0.5, x_test_1D, U_exact_1D, label='Exact Solution 1D')

ax.set_xlabel('x_1')
ax.set_ylabel('x_2')
ax.set_zlabel('u')
ax.set_title('PINN Solution for the Puasson Equation')

color_for_legend = cm.viridis(0.5)  # Get the color from the colormap
proxy = Rectangle((0, 0), 1, 1, fc=color_for_legend, edgecolor="k")
ax.legend([proxy], ['PINN Solution 2D'])
#ax.legend([line, line2, proxy], ['PINN Solution 1D', 'Exact Solution 1D', 'PINN Solution 2D'])

fig.colorbar(surf, shrink=0.5, aspect=5)

plt.show()
