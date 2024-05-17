import h5py

def print_weights(filename):
    with h5py.File(filename, 'r') as f:
        for group in f.keys():
            print(f"Group: {group}")
            for layer in f[group].keys():
                print(f"\tLayer: {layer}")
                for param in f[group][layer].keys():
                    print(f"\t\tParam: {param}")
                    weights = f[group][layer][param][()]
                    print(weights)

filename = 'pinn_wave_equation_model.h5'
print_weights(filename)