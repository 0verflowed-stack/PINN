import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import time
import math

class Puasson1DPINN(tf.keras.Model):
    def __init__(self, layers):
        super(Puasson1DPINN, self).__init__()
        self.hidden_layers = [tf.keras.layers.Dense(layer, activation=tf.nn.tanh) for layer in layers]

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

    def dirichlet_condition(self, model, x_bc_left_right, u_bc_left_right):
        u, _, _ = model.forward(x_bc_left_right)
        return tf.reduce_mean(tf.square(u - u_bc_left_right))

    def loss_fn(self, u_x_x, model, f, x_train, x_bc_left_right, u_bc_left_right):
        puasson_eq = u_x_x - f(x_train)
        bc_dirichlet_left_right = self.dirichlet_condition(model, x_bc_left_right, u_bc_left_right)
        return tf.reduce_mean(tf.square(puasson_eq)) + bc_dirichlet_left_right

# Right-hand side of Puasson equation
def f_1D(x):
    return -np.pi * np.sin(np.pi * x) * np.sin(np.pi * 0.5)#-1

# Function to compute the exact solution
def exact_solution_1D(x):
    return np.sin(np.pi * x)/np.pi#-(np.power(x, 2) - x)/2#np.power(np.pi, 2) * np.sin(np.pi * x2) #(x1*(1-x1))/2

L_x_1D = 1.0 # right side of x

# Define the network, training data, and optimizer
layers_1D = [10, 10, 10, 1]
model_1D = Puasson1DPINN(layers_1D)
x_train_1D = np.linspace(0, L_x_1D, 50)[:, np.newaxis]

optimizer_1D = tf.keras.optimizers.Adam(learning_rate=0.005)

# bc - boundary condition (left and right) dirichlet
#         0 |_______________________| 1
x_bc_left_right_1D = np.array([0, L_x_1D])[:, np.newaxis]
u_bc_left_right_1D = np.array([0, 0])[:, np.newaxis]

# Training loop
n_epochs_1D = 300
loss_array_1D = []
start_time_1D = time.time()

loss_1D = tf.constant(1)
epoch_1D = 0
#for epoch in range(n_epochs + 1):
while loss_1D.numpy() > 0.001:
    with tf.GradientTape() as tape:
        u, u_x, u_x_x = model_1D.forward(x_train_1D)
        loss_1D = model_1D.loss_fn(u_x_x, model_1D, f_1D, x_train_1D, x_bc_left_right_1D, u_bc_left_right_1D)
    grads_1D = tape.gradient(loss_1D, model_1D.trainable_variables)
    optimizer_1D.apply_gradients(zip(grads_1D, model_1D.trainable_variables))

    loss_array_1D.append(loss_1D)
    if epoch_1D % 100 == 0:
        print(f'Epoch {epoch_1D}, Loss: {loss_1D.numpy()}')
    epoch_1D += 1

print("Training took %s seconds" % (time.time() - start_time_1D))

plt.plot(loss_array_1D)
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.grid()
plt.title('Mean loss')
plt.show()

# Compute the exact solution on the test grid
U_exact_1D = exact_solution_1D(x_train_1D)

# Save weights to file
model_1D.save_weights('pinn_wave_equation_model.h5')

# Instantiate a new model with the same layers and wave speed
new_model_1D = Puasson1DPINN(layers_1D)

# Create some dummy input
dummy_x_1D = np.linspace(0, 1, 10)[:, np.newaxis]

# Call the model on the dummy input to create variables
_ = new_model_1D(dummy_x_1D)

# Load the weights from the saved file
loaded_model_1D = new_model_1D.load_weights('pinn_wave_equation_model.h5')

# Visualize the solution
x_test_1D = np.linspace(0, L_x_1D, 100)[:, np.newaxis]

u_pred_1D = new_model_1D.predict(x_test_1D).reshape(x_test_1D.shape[0])

# Compute the exact solution on the test grid
U_exact_1D = exact_solution_1D(x_test_1D)

# Plot of the exact solution
plt.figure()
plt.plot(x_test_1D, U_exact_1D, label='Exact Solution')
plt.xlabel('x')
plt.ylabel('u')
plt.title('Exact Solution for the 1D Poisson Equation')
plt.legend()
plt.show()

# Save and load model (assuming you have a 1D model named 'model_1D')
model_1D.save_weights('pinn_1d_poisson_model.h5')
new_model_1D = Puasson1DPINN(layers_1D)  # Replace with your actual 1D model class and layers
_ = new_model_1D(x_test_1D)  # Dummy call to create variables
new_model_1D.load_weights('pinn_1d_poisson_model.h5')

# Predict the solution using the loaded model
u_pred_1D = new_model_1D.predict(x_test_1D)

# Compute the mean square error between the predicted and exact solutions
mse_1D = np.mean((u_pred_1D - U_exact_1D)**2)
print("Mean Square Error: ", mse_1D)

# Plot of the predicted solution
plt.figure()
plt.plot(x_test_1D, u_pred_1D, label='PINN Solution')
plt.plot(x_test_1D, U_exact_1D, label='Exact Solution')
plt.xlabel('x')
plt.ylabel('u')
plt.title('PINN Solution for the 1D Poisson Equation')
plt.legend()
plt.show()

def calculate_max_relative_error(u_pred, u_exact):
    # Ensure that inputs are 1D arrays
    u_pred = np.squeeze(u_pred)
    u_exact = np.squeeze(u_exact)
    
    # Check for valid dimensions
    if len(u_pred.shape) > 1 or len(u_exact.shape) > 1:
        raise ValueError("Both input arrays should be 1D after squeezing.")
    
    max_relative_error = np.max(np.abs(u_pred - u_exact)) / np.max(np.abs(u_exact))
    error_percentage = 100 * max_relative_error
    return error_percentage

# Example usage:
error_percentage = calculate_max_relative_error(u_pred_1D, U_exact_1D)
print(f"Error based on max difference: {error_percentage:.2f}%")