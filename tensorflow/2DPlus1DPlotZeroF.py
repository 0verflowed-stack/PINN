import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import time
from matplotlib.patches import Rectangle
from matplotlib import cm
from utils import calculate_max_relative_error

tf.random.set_seed(42)
np.random.seed(42)

class Puasson1DPINN(tf.keras.Model):
    def __init__(self, layers, optimizer):
        super(Puasson1DPINN, self).__init__()
        self.hidden_layers = [
            tf.keras.layers.Dense(
                    layer,
                    activation=tf.nn.tanh,
                    kernel_initializer=tf.keras.initializers.GlorotNormal(seed=177013)
                ) for layer in layers
            ]
        self.optimizer = optimizer

    def call(self, x):
        for layer in self.hidden_layers:
            x = layer(x)
        return x

    def forward(self, x):
        x = tf.convert_to_tensor(x, dtype=tf.float32)
        
        with tf.GradientTape(persistent=True) as tape2:
            tape2.watch(x)
            with tf.GradientTape() as tape:
                tape.watch(x)
                u = self.call(x)
            u_x = tape.gradient(u, x)
        u_x_x = tape2.gradient(u_x, x)

        del tape2

        return u, u_x, u_x_x

    def dirichlet_condition(self, x_bc_left_right, u_bc_left_right):
        u, _, _ = self.forward(x_bc_left_right)
        return tf.reduce_mean(tf.square(u - u_bc_left_right))

    def loss_fn(self, u_x_x, x_train, x_bc_left_right, u_bc_left_right):
        puasson_eq = u_x_x - self.f(x_train)
        bc_dirichlet_left_right = self.dirichlet_condition(x_bc_left_right, u_bc_left_right)
        return tf.reduce_mean(tf.square(puasson_eq)) + 10 * bc_dirichlet_left_right

    def train(self, loss_threshold, x_train, x_bc_left_right, u_bc_left_right):
        loss_array = []
        start_time = time.time()

        loss = tf.constant(1)
        epoch = 0

        while loss.numpy() > loss_threshold:
            with tf.GradientTape() as tape:
                u, u_x, u_x_x = self.forward(x_train)
                loss = self.loss_fn(u_x_x, x_train, x_bc_left_right, u_bc_left_right)
            grads = tape.gradient(loss, self.trainable_variables)
            self.optimizer.apply_gradients(zip(grads, self.trainable_variables))

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
        plt.savefig("1d_laplace_equation_training.png")
        plt.show(block=False)

    def f(self, x):
        return 0

    def exact_solution(self, x):
        return -x + 1

N_of_train_points_1D = 5
N_of_test_points_1D = 101
L_x_1D = 0.0
R_x_1D = 1.0
Dirichlet_left_1D = 1
Dirichlet_right_1D = 0
loss_threshold_1D = 0.01
layers_1D = [10, 10, 10, 1]
learning_rate_1D = 0.005
optimizer_1D = tf.keras.optimizers.Adam(learning_rate=learning_rate_1D)

model_1D = Puasson1DPINN(layers_1D, optimizer_1D)

x_train_1D = np.linspace(L_x_1D, R_x_1D, N_of_train_points_1D)[:, np.newaxis]
x_bc_left_right_1D = np.array([L_x_1D, R_x_1D])[:, np.newaxis]
u_bc_left_right_1D = np.array([Dirichlet_left_1D, Dirichlet_right_1D])[:, np.newaxis]

model_1D.train(loss_threshold_1D, x_train_1D, x_bc_left_right_1D, u_bc_left_right_1D)

U_exact_1D = model_1D.exact_solution(x_train_1D)

def save_and_load_weights(model_In, layers, fileName):
    # Save weights to file
    model_In.save_weights(f'{fileName}.h5')

    # Instantiate a new model with the same layers and wave speed
    model_out = Puasson1DPINN(layers)

    # Create some dummy input
    dummy_x_1D = np.linspace(0, 1, 10)[:, np.newaxis]

    # Call the model on the dummy input to create variables
    _ = model_out(dummy_x_1D)

    # Load the weights from the saved file
    loaded_model = model_out.load_weights(f'{fileName}.h5')
    return loaded_model
#save_and_load_weights(model_1D, layers_1D, 'laplace_1d')

# Visualize the solution
x_test_1D = np.linspace(L_x_1D, R_x_1D, N_of_test_points_1D)[:, np.newaxis]
U_exact_1D = model_1D.exact_solution(x_test_1D)
u_pred_1D = model_1D.predict(x_test_1D)

print("Mean Square Error: ", np.mean((u_pred_1D - U_exact_1D)**2))

error_percentage = calculate_max_relative_error(u_pred_1D, U_exact_1D)
print(f"Relative error: {error_percentage:.2f}%")

plt.figure()
plt.plot(x_test_1D, u_pred_1D, label='PINN Solution')
plt.plot(x_test_1D, U_exact_1D, label='Exact Solution')
plt.xlabel('x')
plt.ylabel('u')
plt.title('PINN Solution for the 1D Laplace\'s Equation')
plt.legend()
plt.savefig("1d_laplace_equation_approx_exact_solution.png")
plt.show()

class Puasson2DPINN(tf.keras.Model):
    def __init__(self, layers, optimizer):
        super(Puasson2DPINN, self).__init__()
        self.hidden_layers = [
            tf.keras.layers.Dense(
                    layer,
                    activation=tf.nn.tanh,
                    kernel_initializer=tf.keras.initializers.GlorotNormal(seed=177013)
                ) for layer in layers
            ]
        self.optimizer = optimizer

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

    def dirichlet_condition(self, x1_x2_bc_up, x1_x2_bc_down, u_bc_up, u_bc_down):
        u_up, _, _, _, _ = self.forward(x1_x2_bc_up)
        u_down, _, _, _, _ = self.forward(x1_x2_bc_down)
        return tf.reduce_mean(tf.square(u_up - u_bc_up)) + tf.reduce_mean(tf.square(u_down - u_bc_down))

    def neuman_condition(self, x1_x2_bc_left_right):
        _, u_x1, _,  _, _ = self.forward(x1_x2_bc_left_right)
        return tf.reduce_mean(tf.square(u_x1)) # -∂u/∂x1 + ∂u/∂x1

    def loss_fn(self, x1_x2_train, u_x1_x1, u_x2_x2, x1_x2_bc_left_right, x1_x2_bc_up, x1_x2_bc_down, u_bc_up, u_bc_down):
        puasson_eq = u_x1_x1 + u_x2_x2 - self.f(x1_x2_train)
        bc_dirichlet = self.dirichlet_condition(x1_x2_bc_up, x1_x2_bc_down, u_bc_up, u_bc_down)
        bc_neuman = self.neuman_condition(x1_x2_bc_left_right)

        weight_of_boundaries = 5
        return tf.reduce_mean(tf.square(puasson_eq)) + weight_of_boundaries * (bc_dirichlet + bc_neuman)

    def train(self, loss_threshold, x1_x2_train, x1_x2_bc_left_right, x1_x2_bc_up, x1_x2_bc_down, u_bc_up, u_bc_down):
        loss_array = []
        start_time = time.time()

        loss = tf.constant(1)
        epoch = 0

        while loss.numpy() > loss_threshold:
            with tf.GradientTape() as tape:
                u, u_x1, u_x2, u_x1_x1, u_x2_x2 = self.forward(x1_x2_train)
                loss = self.loss_fn(
                    x1_x2_train, u_x1_x1, u_x2_x2, x1_x2_bc_left_right, x1_x2_bc_up, x1_x2_bc_down, u_bc_up, u_bc_down
                )
            grads = tape.gradient(loss, self.trainable_variables)
            self.optimizer.apply_gradients(zip(grads, self.trainable_variables))

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
        plt.savefig("2d_laplace_equation_training.png")
        plt.show(block=False)

    def f(self, x1_x2):
        x1 = np.array([x[0] for x in x1_x2])
        x2 = np.array([x[1] for x in x1_x2])
        return 0
    
    def exact_solution(self, x1_x2):
        x1 = np.array([x[0] for x in x1_x2])
        return -x1 + 1
    
N_of_train_points_2D = 5
N_of_test_points_2D = 101
L_x1_2D = 0.0
R_x1_2D = 1.0
L_x2_2D = 0.0
R_x2_2D = 1.0
Dirichlet_up_2D = 0
Dirichlet_down_2D = 1
loss_threshold_2D = 0.01
layers_2D = [10, 10, 10, 1]
learning_rate_2D = 0.005
optimizer_2D = tf.keras.optimizers.Adam(learning_rate=learning_rate_2D)

model_2D = Puasson2DPINN(layers_2D, optimizer_2D)

x1_train = np.linspace(L_x1_2D, R_x1_2D, N_of_train_points_2D)[:, np.newaxis]
x2_train = np.linspace(L_x2_2D, R_x2_2D, N_of_train_points_2D)[:, np.newaxis]
x1_mesh, x2_mesh = np.meshgrid(x1_train, x2_train)
x1_x2_train = np.hstack((x1_mesh.flatten()[:, np.newaxis], x2_mesh.flatten()[:, np.newaxis]))

# bc - boundary condition (left and right) neuman
#         __________________________
#         ||                       ||
# [-1, 0] || x2                    || [1, 0] 
#         ||          x1           ||
#         ||_______________________||
# ∂u/∂n = -∂u/∂x1                      ∂u/∂n = ∂u/∂x1
x1_bc_left = L_x1_2D * np.ones_like(x2_train)
x1_bc_right = R_x1_2D * np.ones_like(x2_train)
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
x2_bc_up = R_x2_2D * np.ones_like(x1_train)
x2_bc_down = L_x2_2D * np.ones_like(x1_train)
x1_x2_bc_up = np.hstack((x1_train, x2_bc_up))
x1_x2_bc_down = np.hstack((x1_train, x2_bc_down))

u_bc_up = Dirichlet_up_2D * np.ones_like(x1_train)
u_bc_down = Dirichlet_down_2D * np.ones_like(x1_train)

model_2D.train(
    loss_threshold_2D, x1_x2_train, x1_x2_bc_left_right, x1_x2_bc_up, x1_x2_bc_down, u_bc_up, u_bc_down
)

def save_and_load_model(layers):
    # Save weights to file
    layers.save_weights('pinn_wave_equation_model.h5')
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
    return loaded_model
#save_and_load_model(layers)

# Visualize the solution
x1_test = np.linspace(L_x1_2D, R_x1_2D, N_of_test_points_2D)[:, np.newaxis]
x2_test = np.linspace(L_x2_2D, R_x2_2D, N_of_test_points_2D)[:, np.newaxis]
x1_mesh, x2_mesh = np.meshgrid(x1_test, x2_test)
x1x2_test = np.hstack((x1_mesh.flatten()[:, np.newaxis], x2_mesh.flatten()[:, np.newaxis]))

u_pred_2D = model_2D.predict(x1x2_test).reshape(x1_test.shape[0], x2_test.shape[0])

# Find the index corresponding to x1 = (L_x1_2D + R_x1_2D)/2
index_x1_05 = np.argmin(np.abs(x1_test - (L_x1_2D + R_x1_2D)/2))

# Extract the values along the line x1 = (L_x1_2D + R_x1_2D)/2
u_pred_1D_05 = u_pred_2D[:, index_x1_05]  # Assuming u_pred is 2D with shape (len(x2_test), len(x_test))

# Calculate the MSE for 1D exact and 2D middle of interval [L_x1_2D, R_x1_2D]
mse_1D = np.mean((u_pred_1D_05 - U_exact_1D.squeeze())**2)
print(f"Mean Square Error for 1D exact and 2D middle of interval [{L_x1_2D}, {R_x1_2D}]: {mse_1D}")

error_percentage_2D = calculate_max_relative_error(u_pred_1D_05, U_exact_1D.squeeze())
print(f"Error based on max difference for 1D exact and 2D middle of interval [{L_x1_2D}, {R_x1_2D}]: {error_percentage_2D:.2f}%")

# U_exact_2D = model_2D.exact_solution(x1x2_test)

# error_percentage_2D = calculate_max_relative_error(u_pred_2D.flatten(), U_exact_2D)
# print(f"Error based on max difference for 2D exact and 2D PINN [{L_x1_2D}, {R_x1_2D}]: {error_percentage_2D:.2f}%")


plt.figure(figsize=(8, 6))
plt.pcolor(x1_mesh, x2_mesh, u_pred_2D, cmap='viridis')
plt.colorbar()
plt.xlabel('x_1')
plt.ylabel('x_2')
plt.title('PINN Solution for the Laplace\'s Equation')
plt.savefig("2d_PINN_laplace_equation_solution_heatmap.png")
plt.show(block=False)

fig = plt.figure(figsize=(10, 6))
ax = fig.add_subplot(111, projection='3d')

surf = ax.plot_surface(x1_mesh, x2_mesh, u_pred_2D, cmap='viridis', linewidth=0, antialiased=False, label='PINN Solution 2D', zorder=0, alpha=0.8)
line, = ax.plot(np.ones(N_of_test_points_1D) * 0.5, x_test_1D.flatten(), u_pred_1D.flatten(), label='PINN Solution 1D', zorder=2)
line2, = ax.plot(np.ones(N_of_test_points_1D) * 0.5, x_test_1D.flatten(), U_exact_1D.flatten(), label='Exact Solution 1D', zorder=1)
ax.plot(np.ones(N_of_test_points_1D) * 0.5, x_test_1D.flatten(), U_exact_1D.flatten(), label='Exact Solution 1D')

ax.set_xlabel('x_1')
ax.set_ylabel('x_2')
ax.set_zlabel('u')
ax.set_title('PINN Solution for the Laplace\'s Equation')

color_for_legend = cm.viridis(0.5)
proxy = Rectangle((0, 0), 1, 1, fc=color_for_legend, edgecolor="k")
ax.legend([line, line2, proxy], ['PINN Solution 1D', 'Exact Solution 1D', 'PINN Solution 2D'])

fig.colorbar(surf, shrink=0.5, aspect=5)

plt.savefig("2d_and_1d_compare_PINN_laplace_equation_solution_3D.png")
plt.show()