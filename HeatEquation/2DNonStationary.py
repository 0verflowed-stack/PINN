import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import time
from matplotlib.patches import Rectangle
from matplotlib import cm
from utils import calculate_max_relative_error

#tf.random.set_seed(42)
#np.random.seed(42)

class Puasson1DPINN(tf.keras.Model):
    def __init__(self, layers, optimizer):
        super(Puasson1DPINN, self).__init__()
        self.hidden_layers = [
                tf.keras.layers.Dense(
                    layer,
                    activation=tf.nn.tanh,
                    kernel_initializer=tf.keras.initializers.GlorotNormal()#seed=177013
                ) for layer in layers
            ]
        self.optimizer = optimizer

    def call(self, x_t):
        for layer in self.hidden_layers:
            x_t = layer(x_t)
        return x_t

    def forward(self, x_t):
        x_t = tf.convert_to_tensor(x_t, dtype=tf.float32)
        
        with tf.GradientTape(persistent=True) as tape2:
            tape2.watch(x_t)
            with tf.GradientTape() as tape:
                tape.watch(x_t)
                u = self.call(x_t)
            grads_1 = tape.gradient(u, x_t)
            u_x, u_t = grads_1[:, 0], grads_1[:, 1]
        grads_2 = tape2.gradient(u_x, x_t)
        u_x_x, u_x_t = grads_2[:, 0], grads_2[:, 1]

        del tape2

        return u, u_x, u_t, u_x_x, u_x_t

    def dirichlet_conditions(self, x_t_bc_top_bottom_1D, u_bc_top_bottom_1D):
        u, _, _, _, _ = self.forward(x_t_bc_top_bottom_1D)
        return tf.reduce_mean(tf.square(u - u_bc_top_bottom_1D))
    
    # bc - boundary condition (left and right) neuman
    #         __________________________
    #         ||                       ||
    # [-1, 0] || t                     || [1, 0] 
    #         ||          x            ||
    #         ||_______________________||
    # ∂u/∂n = -∂u/∂x                       ∂u/∂n = ∂u/∂x
    def neuman_conditions(self, x_t_bc_left_1D, x_t_bc_right_1D):
        alpha = 0
        u_left, u_x_left, u_t_left, _, _ = self.forward(x_t_bc_left_1D)
        _, u_x_right, u_t_right, _, _ = self.forward(x_t_bc_right_1D)

        return tf.reduce_mean(tf.square(-u_x_left - alpha * u_left)) + tf.reduce_mean(tf.square(u_x_right - 0))

    def loss_fn(self, u_t, u_x_x, x_t_train, x_t_bc_top_bottom_1D, x_t_bc_left_1D, x_t_bc_right_1D, u_bc_top_bottom_1D):
        a = 1
        puasson_eq = u_t - a * u_x_x - self.f(x_t_train)
        bc_dirichlet_left_right = self.dirichlet_conditions(x_t_bc_top_bottom_1D, u_bc_top_bottom_1D)
        bc_neuman_left_right = self.neuman_conditions(x_t_bc_left_1D, x_t_bc_right_1D)
        return (
                tf.reduce_mean(tf.square(puasson_eq)) +
                bc_dirichlet_left_right + 
                bc_neuman_left_right
            )
    
    def train(self, loss_threshold, x_t_train, x_t_bc_top_bottom_1D, x_t_bc_left_1D, x_t_bc_right_1D, u_bc_top_bottom_1D):
        loss_array = []
        start_time = time.time()
        loss = tf.constant(1)
        epoch = 0

        while loss.numpy() > loss_threshold:
            with tf.GradientTape() as tape:
                u, u_x, u_t, u_x_x, u_x_t = self.forward(x_t_train)
                loss = self.loss_fn(u_t, u_x_x, x_t_train, x_t_bc_top_bottom_1D, x_t_bc_left_1D, x_t_bc_right_1D, u_bc_top_bottom_1D)
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
        #plt.savefig("1d_poisson_equation_training.png")
        plt.show(block=False)

    def f(self, x_t):
        x = np.array([x[0] for x in x_t])
        t = np.array([x[1] for x in x_t])
        return 0#-np.pi * np.sin(np.pi * x)

    def exact_solution(self, x_t):
        x = np.array([x[0] for x in x_t])
        t = np.array([x[1] for x in x_t])
        return np.sin(np.pi * x)/np.pi

# Grid points for x, t and for testing in both directions
N_of_train_points_1D = 100
N_of_train_points_1D_t = 1000
N_of_test_points_1D = 10000
# Define a region
L_x_1D = 0.0
R_x_1D = 1.0
L_t_1D = 0.0
R_t_1D = 1.0
# Define ????????
Dirichlet_top_1D = 1.0 # horizontal axe is X
Dirichlet_bottom_1D = 0.0
# Training stop chriteria
loss_threshold_1D = 0.001
# Define architecture
Layers_1D = [128, 128, 128, 1]
Learning_rate_1D = 0.3
Optimizer_1D = tf.keras.optimizers.AdamW(learning_rate=Learning_rate_1D)

model_1D = Puasson1DPINN(Layers_1D, Optimizer_1D)

x_train_1D = np.linspace(L_x_1D, R_x_1D, N_of_train_points_1D)[:, np.newaxis]
t_train_1D = np.linspace(L_t_1D, R_t_1D, N_of_train_points_1D_t)[:, np.newaxis]
x_mesh_1D, t_mesh_1D = np.meshgrid(x_train_1D, t_train_1D)
x_t_train_1D = np.hstack((x_mesh_1D.flatten()[:, np.newaxis], t_mesh_1D.flatten()[:, np.newaxis]))

x_bc_left_right_1D = np.array([L_x_1D, R_x_1D])[:, np.newaxis]
u_bc_top_bottom_1D = np.array([Dirichlet_bottom_1D, Dirichlet_top_1D])[:, np.newaxis]# Dirichlet boundary condition (left and right)
x_bc_left_right_1D_mesh, t_bc_left_right_1D_mesh = np.meshgrid(x_bc_left_right_1D, u_bc_top_bottom_1D)
x_t_bc_train = np.hstack((x_bc_left_right_1D_mesh.flatten()[:, np.newaxis], t_bc_left_right_1D_mesh.flatten()[:, np.newaxis]))

t_bc_bottom = L_t_1D * np.ones_like(x_train_1D) #REVISE
t_bc_top = R_t_1D * np.ones_like(x_train_1D)
t_bc_bottom_top = np.concatenate((t_bc_bottom, t_bc_top))
x_bc_bottom_top = np.concatenate((x_train_1D, x_train_1D))
x_t_bc_bottom_top = np.hstack((x_bc_bottom_top, t_bc_bottom_top))

x_left = L_x_1D * np.ones_like(t_train_1D) 
x_right = R_x_1D * np.ones_like(t_train_1D)
x_t_bc_left_1D = np.hstack((x_left.flatten()[:, np.newaxis], t_train_1D.flatten()[:, np.newaxis]))
x_t_bc_right_1D = np.hstack((x_right.flatten()[:, np.newaxis], t_train_1D.flatten()[:, np.newaxis]))

u_bc_top_bottom = np.concatenate((Dirichlet_bottom_1D * np.ones_like(x_train_1D) , Dirichlet_top_1D * np.ones_like(x_train_1D)))

model_1D.train(loss_threshold_1D, x_t_train_1D, x_t_bc_bottom_top, x_t_bc_left_1D, x_t_bc_right_1D, u_bc_top_bottom)

def save_and_load_model(model, layers, fileName):
    # Save weights to file
    model.save_weights(f'{fileName}.h5')

    # Instantiate a new model with the same layers and wave speed
    new_model = Puasson1DPINN(layers)

    # Create some dummy input
    dummy_x = np.linspace(0, 1, 10)[:, np.newaxis]

    # Call the model on the dummy input to create variables
    _ = new_model(dummy_x)

    # Load the weights from the saved file
    loaded_model = new_model.load_weights(f'{fileName}.h5')
    return loaded_model

#save_and_load_model(model_1D, layers)

x_test_1D = np.linspace(L_x_1D, R_x_1D, 100)[:, np.newaxis]
t_test_1D = np.linspace(L_t_1D, R_t_1D, 100)[:, np.newaxis]
x_mesh_test_1D, t_mesh_test_1D = np.meshgrid(x_test_1D, t_test_1D)
x_t_test_1D = np.hstack((x_mesh_test_1D.flatten()[:, np.newaxis], t_mesh_test_1D.flatten()[:, np.newaxis]))
u_pred_1D = model_1D.predict(x_t_test_1D).reshape(x_test_1D.shape[0], t_test_1D.shape[0])
U_exact_1D = model_1D.exact_solution(x_t_test_1D).reshape(x_test_1D.shape[0], t_test_1D.shape[0])

print("Mean Square Error: ", np.mean((u_pred_1D - U_exact_1D.reshape(x_test_1D.shape[0], t_test_1D.shape[0]))**2))

error_percentage = calculate_max_relative_error(u_pred_1D, U_exact_1D.reshape(x_test_1D.shape[0], t_test_1D.shape[0]))
print(f"Relative error: {error_percentage:.2f}%")

plt.figure()
plt.figure(figsize=(8, 6))
plt.pcolor(x_mesh_test_1D, t_mesh_test_1D, u_pred_1D, cmap='viridis')
plt.colorbar()
plt.xlabel('x')
plt.ylabel('t')
plt.title('PINN Solution for the Puasson\'s Equation')
#plt.savefig("2d_poisson_equation_exact_solution_heatmap.png")
plt.show()
#plt.plot(x_test_1D, U_exact_1D, label='Exact Solution')
# plt.xlabel('x')
# plt.ylabel('u')
# plt.title('PINN Solution for the 1D Poisson\'s Equation')
# plt.legend()
# plt.savefig("1d_poisson_equation_approx_exact_solution.png")
# plt.show()

# class Puasson2DPINN(tf.keras.Model):
#     def __init__(self, layers, optimizer):
#         super(Puasson2DPINN, self).__init__()
#         self.hidden_layers = [
#                 tf.keras.layers.Dense(
#                     layer,
#                     activation=tf.nn.tanh,
#                     kernel_initializer=tf.keras.initializers.GlorotNormal(seed=177013)
#                 ) for layer in layers
#             ]
#         self.optimizer = optimizer

#     def call(self, x1_x2):
#         for layer in self.hidden_layers:
#             x1_x2 = layer(x1_x2)
#         return x1_x2

#     def forward(self, x1_x2):
#         x1_x2 = tf.convert_to_tensor(x1_x2, dtype=tf.float32)
        
#         with tf.GradientTape(persistent=True) as tape2:
#             tape2.watch(x1_x2)
#             with tf.GradientTape() as tape:
#                 tape.watch(x1_x2)
#                 u = self.call(x1_x2)
#             grads = tape.gradient(u, x1_x2)
#             u_x1, u_x2 = grads[:, 0], grads[:, 1]
#         grads2 = tape2.gradient(u_x1, x1_x2)
#         u_x1_x1 = grads2[:, 0]

#         grads3 = tape2.gradient(u_x2, x1_x2)
#         u_x2_x2 = grads3[:, 1]

#         del tape2

#         return u, u_t, u_x1, u_x2, u_x1_x1, u_x2_x2

#     def dirichlet_condition(self, x1_x2_bc_up_down, u_bc_up, u_bc_down):
#         u, _, _, _, _ = self.forward(x1_x2_bc_up_down)
#         u, _, _, _, _ = self.forward(x1_x2_bc_up_down)
#         return tf.reduce_mean(tf.square(u))

#     def neuman_condition(self, x1_x2_bc_left_right):
#         _, u_x1, _,  _, _ = self.forward(x1_x2_bc_left_right)
#         return tf.reduce_mean(tf.square(u_x1)) # -∂u/∂x1 + ∂u/∂x1

#     def loss_fn(self, x1_x2_train, u_t, u_x1_x1, u_x2_x2, x1_x2_bc_left_right, x1_x2_bc_up_down, u_bc_up, u_bc_down):
#         puasson_eq = u_t - u_x1_x1 - u_x2_x2 - self.f(x1_x2_train)
#         bc_dirichlet = self.dirichlet_condition(x1_x2_bc_up_down, u_bc_up, u_bc_down)
#         bc_neuman = self.neuman_condition(x1_x2_bc_left_right)

#         return tf.reduce_mean(tf.square(puasson_eq)) + 2 * (bc_dirichlet + bc_neuman)

#     def train(self, loss_threshold, x1_x2_train, x1_x2_bc_left_right, x1_x2_bc_up_down, u_bc_up, u_bc_down):
#         loss_array = []
#         start_time = time.time()

#         loss = tf.constant(1)
#         epoch = 0
#         while loss.numpy() > loss_threshold:
#             with tf.GradientTape() as tape:
#                 u, u_t, u_x1, u_x2, u_x1_x1, u_x2_x2 = self.forward(x1_x2_train)
#                 loss = self.loss_fn(x1_x2_train, u_t, u_x1_x1, u_x2_x2, x1_x2_bc_left_right, x1_x2_bc_up_down, u_bc_up, u_bc_down)
#             grads = tape.gradient(loss, self.trainable_variables)
#             self.optimizer.apply_gradients(zip(grads, self.trainable_variables))

#             loss_array.append(loss)
#             if epoch % 100 == 0:
#                 print(f'Epoch {epoch}, Loss: {loss.numpy()}')
#             epoch += 1

#         print("Training took %s seconds" % (time.time() - start_time))
#         print(f"Last epoch: {epoch}, loss: {loss.numpy()}")

#         plt.plot(loss_array)
#         plt.xlabel('Epoch')
#         plt.ylabel('Loss')
#         plt.grid()
#         plt.title('Mean loss')
#         plt.savefig("2d_poisson_equation_training.png")
#         plt.show(block=False)

#     def f(self, x1_x2):
#         x1 = np.array([x[0] for x in x1_x2])
#         x2 = np.array([x[1] for x in x1_x2])
#         return -np.pi * np.sin(np.pi * x1) * np.sin(np.pi * x2)

# N_of_train_points_2D = 15
# N_of_test_points_2D = 101
# L_x1_2D = 0.0
# R_x1_2D = 1.0
# L_x2_2D = 0.0
# R_x2_2D = 1.0
# Dirichlet_up_2D = 0.0
# Dirichlet_down_2D = 0.0
# loss_threshold_2D = 0.01
# Layers_2D = [10, 10, 10, 1]
# Learning_rate_2D = 0.005
# Optimizer_2D = tf.keras.optimizers.Adam(learning_rate=Learning_rate_2D)


# model_2D = Puasson2DPINN(Layers_2D, Optimizer_2D)
# x1_train = np.linspace(L_x1_2D, R_x1_2D, N_of_train_points_2D)[:, np.newaxis]
# x2_train = np.linspace(L_x2_2D, R_x2_2D, N_of_train_points_2D)[:, np.newaxis]
# x1_mesh, x2_mesh = np.meshgrid(x1_train, x2_train)
# x1_x2_train = np.hstack((x1_mesh.flatten()[:, np.newaxis], x2_mesh.flatten()[:, np.newaxis]))

# # bc - boundary condition (left and right) neuman
# #         __________________________
# #         ||                       ||
# # [-1, 0] || x2                    || [1, 0] 
# #         ||          x1           ||
# #         ||_______________________||
# # ∂u/∂n = -∂u/∂x1                      ∂u/∂n = ∂u/∂x1
# x1_bc_left = L_x1_2D * np.ones_like(x2_train)
# x1_bc_right = R_x1_2D * np.ones_like(x2_train)
# x1_bc_left_right = np.concatenate((x1_bc_left, x1_bc_right))
# t_bc_bottom_top = np.concatenate((x2_train, x2_train))
# x1_x2_bc_bottom_top = np.hstack((x1_bc_left_right, t_bc_bottom_top))

# # bc - boundary condition (up and down) dirichlet
# #          [0, 1]  ∂u/∂n = ∂u/∂x2
# #  =======================
# # |                       |
# # | x2                    |
# # |          x1           |
# # |=======================|
# #          [0, -1]  ∂u/∂n = -∂u/∂x2
# x2_bc_up = L_x1_2D * np.ones_like(x1_train)
# x2_bc_down = R_x1_2D * np.ones_like(x1_train)
# x2_bc_up_down = np.concatenate((x2_bc_up, x2_bc_down))
# x1_bc_up_down = np.concatenate((x1_train, x1_train))
# x1_x2_bc_up_down = np.hstack((x1_bc_up_down, x2_bc_up_down))

# u_bc_up = Dirichlet_up_2D * np.ones_like(x1_train)
# u_bc_down = Dirichlet_down_2D * np.ones_like(x1_train)

# model_2D.train(loss_threshold_2D, x1_x2_train, x1_x2_bc_bottom_top, x1_x2_bc_up_down, u_bc_up, u_bc_down)

# def save_and_load_model_2D(model, layers, fileName):
#     # Save weights to file
#     model.save_weights(f'{fileName}.h5')

#     # Instantiate a new model with the same layers and wave speed
#     new_model = Puasson2DPINN(layers)

#     # Create some dummy input
#     dummy_x = np.linspace(0, 1, 10)
#     dummy_t = np.linspace(0, 1, 10)
#     dummy_xt = np.array([[x1, x2] for x1 in dummy_x for x2 in dummy_t], dtype=np.float32)

#     # Call the model on the dummy input to create variables
#     _ = new_model(dummy_xt)

#     # Load the weights from the saved file
#     loaded_model = new_model.load_weights(f'{fileName}.h5')
#     return loaded_model
# #save_and_load_model_2D(model, layers)

# # Visualize the solution
# x1_test = np.linspace(L_x1_2D, R_x1_2D, N_of_test_points_2D)[:, np.newaxis]
# x2_test = np.linspace(L_x2_2D, R_x2_2D, N_of_test_points_2D)[:, np.newaxis]
# x1_mesh, x2_mesh = np.meshgrid(x1_test, x2_test)
# x1x2_test = np.hstack((x1_mesh.flatten()[:, np.newaxis], x2_mesh.flatten()[:, np.newaxis]))

# u_pred = model_2D.predict(x1x2_test).reshape(x1_test.shape[0], x2_test.shape[0])

# plt.figure(figsize=(8, 6))
# plt.pcolor(x1_mesh, x2_mesh, u_pred, cmap='viridis')
# plt.colorbar()
# plt.xlabel('x_1')
# plt.ylabel('x_2')
# plt.title('PINN Solution for the Puasson\'s Equation')
# plt.savefig("2d_poisson_equation_exact_solution_heatmap.png")
# plt.show(block=False)

# fig = plt.figure(figsize=(10, 6))
# ax = fig.add_subplot(111, projection='3d')
# # line, = ax.plot(np.ones_like(x_test_1D) * 0.5, x_test_1D, u_pred_1D, label='PINN Solution 1D', zorder=2)
# # line2, = ax.plot(np.ones_like(x_test_1D) * 0.5, x_test_1D, U_exact_1D, label='Exact Solution 1D', zorder=1)
# surf = ax.plot_surface(x1_mesh, x2_mesh, u_pred, cmap='viridis', linewidth=0, antialiased=False, label='PINN Solution 2D', zorder=0)
# #ax.plot(np.ones_like(x_test_1D) * 0.5, x_test_1D, U_exact_1D, label='Exact Solution 1D')
# ax.set_xlabel('x_1')
# ax.set_ylabel('x_2')
# ax.set_zlabel('u')
# ax.set_title('PINN Solution for the Puasson\'s Equation')
# color_for_legend = cm.viridis(0.5)  # Get the color from the colormap
# proxy = Rectangle((0, 0), 1, 1, fc=color_for_legend, edgecolor="k")
# ax.legend([proxy], ['PINN Solution 2D'])
# #ax.legend([line, line2, proxy], ['PINN Solution 1D', 'Exact Solution 1D', 'PINN Solution 2D'])
# fig.colorbar(surf, shrink=0.5, aspect=5)
# plt.savefig("2d_poisson_equation_approx_solution.png")
# plt.show()