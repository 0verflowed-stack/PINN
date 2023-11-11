import time
import math
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from mplcursors import cursor
import networkx as nx

# np.random.seed(177013)
# tf.random.set_seed(177013)


class PINN(tf.keras.Model):
    def __init__(self, layers):
        super(PINN, self).__init__()
        self.hidden_layers = [
            tf.keras.layers.Dense(layer, activation=tf.nn.tanh) for layer in layers
            ]
        
    # def build(self, input_shape):
    #     self.w = self.add_weight(shape=(input_shape[-1], 10),
    #                              initializer='random_normal',
    #                              trainable=True)
    #     self.b = self.add_weight(shape=(10,),
    #                              initializer='zeros',
    #                              trainable=True)

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
layers = [5, 5, 5, 1] # NN architecture
model = PINN(layers)
# model.build([1])
x_train = np.linspace(0, 1, 100)[:, np.newaxis] # unflatten each value to array
optimizer = tf.keras.optimizers.Adam(learning_rate=0.005)


n_epochs = 1000
loss_array = [] 
best_epoch = 0
best_loss = float('inf')

start_time = time.time()
loss = tf.constant(1)
epoch = 0
#for epoch in range(n_epochs + 1):
while loss.numpy() > math.pow(0.1, 3):
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
    epoch += 1

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



def visualize_weights(model):
    for i, layer in enumerate(model.layers):
        weights = layer.get_weights()
        if len(weights) > 0:
            w = weights[0]  # Get the weights of the layer (ignoring biases)
            fig, ax = plt.subplots()
            if len(w.shape) == 2:  # Dense layer
                # Create a heatmap
                cax = ax.imshow(w, cmap="viridis")
                ax.set_title(f"Layer {i}: {layer.name}")
                fig.colorbar(cax)
            elif len(w.shape) == 1:  # 1D weights (e.g., embeddings)
                ax.bar(np.arange(len(w)), w)
                ax.set_title(f"Layer {i}: {layer.name}")
            else:
                print(f"Layer {i} has unsupported weight shape {w.shape}.")
            plt.show()

# Replace 'your_keras_model' with your actual Keras model
visualize_weights(model)

def visualize_weighted_graph(model):
    for l, layer in enumerate(model.hidden_layers):  # loop over hidden layers only
        W = layer.get_weights()[0]  # Extract weights (ignoring biases)
        G = nx.DiGraph()  # Create a directed graph object

        n_neurons_in = W.shape[0]  # Neurons in the layer (input)
        n_neurons_out = W.shape[1]  # Neurons in the next layer (output)

        # Add nodes
        for i in range(n_neurons_in):
            G.add_node(f"Layer_{l}_Neuron_{i}")

        for j in range(n_neurons_out):
            G.add_node(f"Layer_{l+1}_Neuron_{j}")

        # Add edges with weights
        for i in range(n_neurons_in):
            for j in range(n_neurons_out):
                weight = W[i, j]
                G.add_edge(f"Layer_{l}_Neuron_{i}", f"Layer_{l+1}_Neuron_{j}", weight=weight)

        # Draw the graph
        pos = nx.spring_layout(G)  # positions for all nodes
        nx.draw(G, pos, with_labels=True, node_color='skyblue', font_size=10, node_size=700)
        labels = nx.get_edge_attributes(G, 'weight')
        nx.draw_networkx_edge_labels(G, pos, edge_labels=labels)

        plt.title(f"Weights for hidden layer {l+1}")
        plt.show()

visualize_weighted_graph(model)


def visualize_all_layers(model):
    G = nx.DiGraph()  # Create a single directed graph object for all layers
    pos = {}  # Dictionary to store the positions of nodes

    y_offset = 0  # Initialize y_offset to align nodes

    # Loop over hidden layers
    for l, layer in enumerate(model.hidden_layers):
        W = layer.get_weights()[0]  # Extract weights (ignoring biases)
        
        n_neurons_in = W.shape[0]  # Neurons in this layer (input)
        n_neurons_out = W.shape[1]  # Neurons in the next layer (output)

        # Add nodes and set their positions
        for i in range(n_neurons_in):
            node_name = f"L{l}N{i}"
            G.add_node(node_name)
            pos[node_name] = (l, y_offset - i * 10)  # x is layer index, y is neuron index
            
        for j in range(n_neurons_out):
            node_name = f"L{l+1}N{j}"
            G.add_node(node_name)
            pos[node_name] = (l+1, y_offset - j * 10)  # x is layer index, y is neuron index

        # Adjust y_offset for next layer
        y_offset -= max(n_neurons_in, n_neurons_out) + 1

        # Add edges with weights
        for i in range(n_neurons_in):
            for j in range(n_neurons_out):
                weight = W[i, j]
                G.add_edge(f"L{l}N{i}", f"L{l+1}N{j}", weight=weight)

    # Draw the graph
    nx.draw(G, pos, with_labels=True, node_color='skyblue', font_size=10, node_size=700)
    labels = nx.get_edge_attributes(G, 'weight')
    nx.draw_networkx_edge_labels(G, pos, edge_labels=labels)

    plt.title("Weighted Graph for All Hidden Layers")
    plt.show()

# Call this function to visualize the weights as a single graph
visualize_all_layers(model)