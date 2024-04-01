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

    def forward(self, xt):
        xt = tf.convert_to_tensor(xt, dtype=tf.float32)
        
        with tf.GradientTape(persistent=True) as tape2:
            tape2.watch(xt)
            with tf.GradientTape() as tape:
                tape.watch(xt)
                u = self.call(xt)
            grads = tape.gradient(u, xt)
            u_x, u_t = grads[:, 0], grads[:, 1]
        grads2 = tape2.gradient(u_x, xt)
        u_xx = grads2[:, 0]

        grads3 = tape2.gradient(u_t, xt)
        u_tt = grads3[:, 1]

        del tape2

        return u, u_x, u_xx, u_tt

def initial_conditions(model, xt, f, g):
    u, _, _, u_t = model.forward(xt)
    u_0 = u - f(xt[:, 0])
    u_t_0 = u_t - g(xt[:, 0])
    return tf.reduce_mean(tf.square(u_0)) +tf.reduce_mean(tf.square(u_t_0))

def boundary_conditions(model, xt):
    u, _, _, _ = model.forward(xt)
    return tf.reduce_mean(tf.square(u))

def loss_fn(u, u_x, u_xx, u_tt, c, model, xt_ic, f, g, xt_bc):
    wave_eq = u_tt - c**2 * u_xx
    ic_loss = initial_conditions(model, xt_ic, f, g)
    bc_loss = boundary_conditions(model, xt_bc)
    return tf.reduce_mean(tf.square(wave_eq)) + ic_loss + bc_loss

# Define the initial and boundary conditions as functions
def f(x):
    return np.ones_like(x)

def g(x):
    return np.zeros_like(x)

# Function to compute the exact solution
def exact_solution(x, t, c):
    return np.sin(np.pi * x) * np.cos(np.pi * c * t)

print("Available devices:", tf.config.list_physical_devices())

L = 1.0 # right side of x
c = 1.0 # wave velocity

# Define the network, training data, and optimizer
layers = [2, 50, 50, 50, 1]
model = WaveEquationPINN(layers)
x_train = np.linspace(0, L, 100)[:, np.newaxis]
t_train = np.linspace(0, 1, 100)[:, np.newaxis]
x_mesh, t_mesh = np.meshgrid(x_train, t_train)
xt_train = np.hstack((x_mesh.flatten()[:, np.newaxis], t_mesh.flatten()[:, np.newaxis]))

optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)

# Modify the training data to include initial and boundary conditions
# ic - initial condition
x_ic = np.linspace(0, L, 100)[:, np.newaxis]
t_ic = np.zeros_like(x_ic)
xt_ic = np.hstack((x_ic, t_ic))

# bc - boundary condition
x_bc_left = np.zeros_like(t_train)
x_bc_right = L * np.ones_like(t_train)
x_bc = np.concatenate((x_bc_left, x_bc_right))
t_bc = np.concatenate((t_train, t_train))
xt_bc = np.hstack((x_bc, t_bc))

# Training loop
n_epochs = 1000
#batch_size = len(xt_train) #1024 * 2
#n_batches = len(xt_train) // batch_size
#print("n_batches", n_batches)
loss_array = []
start_time = time.time()

for epoch in range(n_epochs):
    with tf.GradientTape() as tape:
        u, u_x, u_xx, u_tt = model.forward(xt_train)
        loss = loss_fn(u, u_x, u_xx, u_tt, c, model, xt_ic, f, g, xt_bc)
    grads = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(grads, model.trainable_variables))

    loss_array.append(loss)
    #print(f'Epoch {epoch}, Loss: {loss.numpy()}')
    if epoch % 100 == 0:
        print(f'Epoch {epoch}, Loss: {loss.numpy()}')

# for epoch in range(n_epochs):
#     # Shuffle the data
#     indices = np.random.permutation(len(xt_train))
#     xt_train_shuffled = xt_train[indices]

#     for batch in range(n_batches + 1):  # Add 1 to account for the last batch
#         # Get the batch data
#         #print(f"Start batch {batch}")
#         start = batch * batch_size
#         end = min((batch + 1) * batch_size, len(xt_train))  # Use min to avoid going over the length
#         xt_batch = xt_train_shuffled[start:end]

#         # Train the model on the batch
#         with tf.GradientTape() as tape:
#             u, u_x, u_xx, u_tt = model.forward(xt_batch)
#             loss = loss_fn(u, u_x, u_xx, u_tt, c, model, xt_ic, f, g, xt_bc)
#         grads = tape.gradient(loss, model.trainable_variables)
#         optimizer.apply_gradients(zip(grads, model.trainable_variables))
#         #print(f"End batch {batch}")

#     # Compute the total loss after each epoch
#     u, u_x, u_xx, u_tt = model.forward(xt_train)
#     total_loss = loss_fn(u, u_x, u_xx, u_tt, c, model, xt_ic, f, g, xt_bc)
#     loss_array.append(total_loss)
#     print(f'Epoch {epoch}, Loss: {total_loss.numpy()}')
#     if epoch % 100 == 0:
#         print(f'Epoch {epoch}, Loss: {total_loss.numpy()}')

print("Training took %s seconds" % (time.time() - start_time))

plt.plot(loss_array)
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.grid()
plt.title('Mean loss')
plt.show(block=False)

# Compute the exact solution on the test grid
U_exact = exact_solution(x_mesh, t_mesh, c)

# 3D plot of the exact solution
fig_exact = plt.figure()
ax_exact = fig_exact.add_subplot(111, projection='3d')
ax_exact.plot_surface(x_mesh, t_mesh, U_exact, cmap='viridis')
ax_exact.set_xlabel('x')
ax_exact.set_ylabel('t')
ax_exact.set_zlabel('u')
ax_exact.set_title('Exact Solution for the Wave Equation')
plt.show(block=False)

# Save weights to file
model.save_weights('pinn_wave_equation_model.h5')

# Instantiate a new model with the same layers and wave speed
new_model = WaveEquationPINN(layers)

# Create some dummy input
dummy_x = np.linspace(0, 1, 10)
dummy_t = np.linspace(0, 1, 10)
dummy_xt = np.array([[x, t] for x in dummy_x for t in dummy_t], dtype=np.float32)

# Call the model on the dummy input to create variables
_ = new_model(dummy_xt)

# Load the weights from the saved file
loaded_model = new_model.load_weights('pinn_wave_equation_model.h5')

# Visualize the solution
x_test = np.linspace(0, L, 1000)[:, np.newaxis]
t_test = np.linspace(0, 1, 1000)[:, np.newaxis]
x_mesh, t_mesh = np.meshgrid(x_test, t_test)
xt_test = np.hstack((x_mesh.flatten()[:, np.newaxis], t_mesh.flatten()[:, np.newaxis]))

u_pred = new_model.predict(xt_test).reshape(x_test.shape[0], t_test.shape[0])

# Compute the exact solution for the test points
u_exact = exact_solution(x_mesh, t_mesh, c)

# Compute the mean square error between the predicted solution and the exact solution 
mse = np.mean((u_pred - u_exact)**2)
print("Mean Square Error: ", mse)

plt.figure(figsize=(8, 6))
plt.pcolor(x_mesh, t_mesh, u_pred, cmap='viridis')
plt.colorbar()
plt.xlabel('x')
plt.ylabel('t')
plt.title('PINN Solution for the Wave Equation')
plt.show(block=False)


fig = plt.figure(figsize=(10, 6))
ax = fig.add_subplot(111, projection='3d')

surf = ax.plot_surface(x_mesh, t_mesh, u_pred, cmap='viridis', linewidth=0, antialiased=False)

ax.set_xlabel('x')
ax.set_ylabel('t')
ax.set_zlabel('u')
ax.set_title('PINN Solution for the Wave Equation')

fig.colorbar(surf, shrink=0.5, aspect=5)

plt.show()