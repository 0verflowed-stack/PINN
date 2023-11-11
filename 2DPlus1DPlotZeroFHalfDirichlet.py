import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import time
from matplotlib.patches import Rectangle
from matplotlib import cm
from utils import calculate_max_relative_error

tf.random.set_seed(42)
np.random.seed(42)

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

    def dirichlet_condition(
            self, x1_x2_bc_up_1, x1_x2_bc_up_2, x1_x2_bc_down_1, x1_x2_bc_down_2,
            u_bc_up_1, u_bc_up_2, u_bc_down_1, u_bc_down_2
        ):
        u_up_1, _, _, _, _ = self.forward(x1_x2_bc_up_1)
        u_up_2, _, _, _, _ = self.forward(x1_x2_bc_up_2)
        u_down_1, _, _, _, _ = self.forward(x1_x2_bc_down_1)
        u_down_2, _, _, _, _ = self.forward(x1_x2_bc_down_2)
        return tf.reduce_mean(tf.square(u_up_1 - u_bc_up_1)) + \
            tf.reduce_mean(tf.square(u_up_2 - u_bc_up_2)) + \
            tf.reduce_mean(tf.square(u_down_1 - u_bc_down_1)) + \
            tf.reduce_mean(tf.square(u_down_2 - u_bc_down_2))

    def neuman_condition(self, x1_x2_bc_left_right):
        _, u_x1, _,  _, _ = self.forward(x1_x2_bc_left_right)
        return tf.reduce_mean(tf.square(u_x1)) # -∂u/∂x1 + ∂u/∂x1

    def loss_fn(
            self, x1_x2_train, u_x1_x1, u_x2_x2, x1_x2_bc_left_right, x1_x2_bc_up_1, x1_x2_bc_up_2,
            x1_x2_bc_down_1, x1_x2_bc_down_2, u_bc_up_1, u_bc_up_2, u_bc_down_1, u_bc_down_2
        ):
        puasson_eq = u_x1_x1 + u_x2_x2 - self.f(x1_x2_train)
        bc_dirichlet = self.dirichlet_condition(
            x1_x2_bc_up_1, x1_x2_bc_up_2, x1_x2_bc_down_1, x1_x2_bc_down_2,
            u_bc_up_1, u_bc_up_2, u_bc_down_1, u_bc_down_2
        )
        bc_neuman = self.neuman_condition(x1_x2_bc_left_right)

        return tf.reduce_mean(tf.square(puasson_eq)) + 5 * (bc_dirichlet + bc_neuman)

    def train(
            self, loss_threshold, x1_x2_train, x1_x2_bc_left_right, x1_x2_bc_up_1, x1_x2_bc_up_2, x1_x2_bc_down_1,
            x1_x2_bc_down_2, u_bc_up_1, u_bc_up_2, u_bc_down_1, u_bc_down_2
        ):
        loss_array = []
        start_time = time.time()

        loss = tf.constant(1)
        epoch = 0
        while loss.numpy() > loss_threshold:
            with tf.GradientTape() as tape:
                u_fdm, u_x1, u_x2, u_x1_x1, u_x2_x2 = self.forward(x1_x2_train)
                loss = self.loss_fn(
                        x1_x2_train, u_x1_x1, u_x2_x2, x1_x2_bc_left_right, x1_x2_bc_up_1, x1_x2_bc_up_2, x1_x2_bc_down_1,
                        x1_x2_bc_down_2, u_bc_up_1, u_bc_up_2, u_bc_down_1, u_bc_down_2
                    )
            grads = tape.gradient(loss, self.trainable_variables)
            optimizer.apply_gradients(zip(grads, self.trainable_variables))

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
        plt.savefig('2D_PINN_half_dirichlet_Laplace_training.png')
        plt.show(block=False)

    def f(self, x1_x2):
        x1 = np.array([x[0] for x in x1_x2])
        x2 = np.array([x[1] for x in x1_x2])
        return 1

N_of_train_points = 11
N_of_test_points = 201
L_x1 = 0.0
R_x1 = 1.0
L_x2 = 0.0
R_x2 = 1.0
Dirichlet_up_left = 0
Dirichlet_up_right = 1
Dirichlet_down_left = 1
Dirichlet_down_right = 0
loss_threshold = 0.5
layers = [20, 20, 20, 1]
learning_rate = 0.005
optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)

model_2D = Puasson2DPINN(layers, optimizer)

x1_train = np.linspace(L_x1, R_x1, N_of_train_points)[:, np.newaxis]
x2_train = np.linspace(L_x2, R_x2, N_of_train_points)[:, np.newaxis]
x1_mesh, x2_mesh = np.meshgrid(x1_train, x2_train)
x1_x2_train = np.hstack((x1_mesh.flatten()[:, np.newaxis], x2_mesh.flatten()[:, np.newaxis]))

# bc - boundary condition (left and right) neuman
#         __________________________
#         ||                       ||
# [-1, 0] || x2                    || [1, 0] 
#         ||          x1           ||
#         ||_______________________||
# ∂u/∂n = -∂u/∂x1                      ∂u/∂n = ∂u/∂x1
x1_bc_left = L_x1 * np.ones_like(x2_train)
x1_bc_right = R_x1 * np.ones_like(x2_train)
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
x2_bc_up_1 = R_x2 * np.ones_like(x1_train)[:len(x1_train)//2]
x2_bc_up_2 = R_x2 * np.ones_like(x1_train)[len(x1_train)//2:]
x2_bc_down_1 = L_x2 * np.ones_like(x1_train)[:len(x1_train)//2]
x2_bc_down_2 = L_x2 * np.ones_like(x1_train)[len(x1_train)//2:]
x1_x2_bc_up_1 = np.hstack((x1_train[:len(x1_train)//2], x2_bc_up_1))
x1_x2_bc_up_2 = np.hstack((x1_train[len(x1_train)//2:], x2_bc_up_2))
x1_x2_bc_down_1 = np.hstack((x1_train[:len(x1_train)//2], x2_bc_down_1))
x1_x2_bc_down_2 = np.hstack((x1_train[len(x1_train)//2:], x2_bc_down_2))

u_bc_up_1 = Dirichlet_up_left * np.ones_like(x1_train)[:len(x1_train)//2]
u_bc_up_2 = Dirichlet_up_right * np.ones_like(x1_train)[len(x1_train)//2:]
u_bc_down_1 = Dirichlet_down_left * np.ones_like(x1_train)[:len(x1_train)//2]
u_bc_down_2 = Dirichlet_down_right * np.ones_like(x1_train)[len(x1_train)//2:]

model_2D.train(
    loss_threshold, x1_x2_train, x1_x2_bc_left_right, x1_x2_bc_up_1, x1_x2_bc_up_2, x1_x2_bc_down_1,
    x1_x2_bc_down_2, u_bc_up_1, u_bc_up_2, u_bc_down_1, u_bc_down_2
)

# Visualize the solution
x_test = np.linspace(L_x1, R_x1, N_of_test_points)[:, np.newaxis]
x2_test = np.linspace(L_x2, R_x2, N_of_test_points)[:, np.newaxis]
x1_mesh, x2_mesh = np.meshgrid(x_test, x2_test)
x1x2_test = np.hstack((x1_mesh.flatten()[:, np.newaxis], x2_mesh.flatten()[:, np.newaxis]))

u_pred = model_2D.predict(x1x2_test).reshape(x_test.shape[0], x2_test.shape[0])

plt.figure(figsize=(8, 6))
plt.pcolor(x1_mesh, x2_mesh, u_pred, cmap='viridis')
plt.colorbar()
plt.xlabel('x_1')
plt.ylabel('x_2')
plt.title('PINN Solution for the Laplace\'s Equation')
plt.savefig('PINN_solution_half_dirichlet_Laplace_3D_heatmap.png')
plt.show(block=False)

fig = plt.figure(figsize=(10, 6))
ax = fig.add_subplot(111, projection='3d')
surf = ax.plot_surface(x1_mesh, x2_mesh, u_pred, cmap='viridis', linewidth=0, antialiased=False, label='PINN Solution 2D')
ax.set_xlabel('x_1')
ax.set_ylabel('x_2')
ax.set_zlabel('u')
ax.set_title('PINN Solution for the Laplace\'s Equation')

color_for_legend = cm.viridis(0.5) # Get the color from the colormap
proxy = Rectangle((0, 0), 1, 1, fc=color_for_legend, edgecolor="k")
ax.legend([proxy], ['PINN Solution 2D'])
fig.colorbar(surf, shrink=0.5, aspect=5)
plt.savefig('PINN_solution_half_dirichlet_Laplace_3D.png')
plt.show()

print("Solving useing FEM")
from fenics import *
import matplotlib.pyplot as plt
import numpy as np

# Create mesh and define function space
nx, ny = N_of_train_points, N_of_train_points  # number of nodes in the x and y directions
mesh = UnitSquareMesh(101, 101)
V = FunctionSpace(mesh, 'P', 1)

# Define boundary conditions
def boundary_upper_left(x, on_boundary):
    return on_boundary and near(x[1], R_x1) and x[0] <= 0.5

def boundary_upper_right(x, on_boundary):
    return on_boundary and near(x[1], R_x1) and x[0] > 0.5

def boundary_lower_left(x, on_boundary):
    return on_boundary and near(x[1], L_x1) and x[0] <= 0.5

def boundary_lower_right(x, on_boundary):
    return on_boundary and near(x[1], L_x1) and x[0] > 0.5

bc_upper_left = DirichletBC(V, Constant(Dirichlet_up_left), boundary_upper_left)
bc_upper_right = DirichletBC(V, Constant(Dirichlet_up_right), boundary_upper_right)
bc_lower_left = DirichletBC(V, Constant(Dirichlet_down_left), boundary_lower_left)
bc_lower_right = DirichletBC(V, Constant(Dirichlet_down_right), boundary_lower_right)

bcs = [bc_upper_left, bc_upper_right, bc_lower_left, bc_lower_right]

# Define problem
u = TrialFunction(V)
v = TestFunction(V)
f = Constant(1)  # source term
a = -dot(grad(u), grad(v))*dx
L = f*v*dx

# Compute solution
u = Function(V)
solve(a == L, u, bcs)

# Plot solution using matplotlib
c = plot(u)
plt.colorbar(c)
plt.title("FEM solution of Laplace\'s equation")
plt.xlabel('x')
plt.ylabel('y')
plt.savefig('fenics_solution_half_dirichlet_Laplace_heatmap.png')
plt.show()

# Create a 3D plot
fig = plt.figure(figsize=(8, 6))
ax = fig.add_subplot(111, projection='3d')

x = np.linspace(L_x1, R_x1, N_of_test_points)
y = np.linspace(L_x2, R_x2, N_of_test_points)
X, Y = np.meshgrid(x, y)
points = np.vstack((X.ravel(), Y.ravel())).T

# Initialize an array to store the solution
u_fenics = np.zeros_like(points[:, 0])

# Evaluate the FEniCS solution at each point
for i, point in enumerate(points):
    u_fenics[i] = u(point)

# Reshape the solution to a 2D grid for plotting or comparison
u_fenics_grid = u_fenics.reshape((len(x), len(y)))

surf = ax.plot_surface(X, Y, u_fenics_grid, cmap='viridis', edgecolor='k')
plt.colorbar(surf)
plt.title("3D plot of the solution")
plt.xlabel('x')
plt.ylabel('y')
ax.set_zlabel('u')
plt.savefig('fenics_solution_half_dirichlet_Laplace_3D.png')
plt.show()

print("Mean Squared Error (PINN and FEM):", np.mean((u_fenics_grid - u_pred)**2))

relative_error = calculate_max_relative_error(u_fenics_grid, u_pred)
print(f"Relative error (PINN and FEM): {relative_error:.2f}%")