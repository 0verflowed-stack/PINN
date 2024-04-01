from fenics import *
import matplotlib.pyplot as plt
import numpy as np

# Create mesh and define function space
nx, ny = 101, 101  # number of nodes in the x and y directions
mesh = UnitSquareMesh(nx, ny)
V = FunctionSpace(mesh, 'P', 1)

# Define boundary conditions
def boundary_upper_left(x, on_boundary):
    return on_boundary and near(x[1], 1) and x[0] <= 0.5

def boundary_upper_right(x, on_boundary):
    return on_boundary and near(x[1], 1) and x[0] > 0.5

def boundary_lower_left(x, on_boundary):
    return on_boundary and near(x[1], 0) and x[0] <= 0.5

def boundary_lower_right(x, on_boundary):
    return on_boundary and near(x[1], 0) and x[0] > 0.5

bc_upper_left = DirichletBC(V, Constant(0), boundary_upper_left)
bc_upper_right = DirichletBC(V, Constant(1), boundary_upper_right)
bc_lower_left = DirichletBC(V, Constant(1), boundary_lower_left)
bc_lower_right = DirichletBC(V, Constant(0), boundary_lower_right)

bcs = [bc_upper_left, bc_upper_right, bc_lower_left, bc_lower_right]

# Define problem
u = TrialFunction(V)
v = TestFunction(V)
f = Constant(0)  # source term
a = dot(grad(u), grad(v))*dx
L = f*v*dx

# Compute solution
u = Function(V)
solve(a == L, u, bcs)

x = np.linspace(0, 1, 101)  # Change the range and number of points as needed
y = np.linspace(0, 1, 101)
X, Y = np.meshgrid(x, y)
points = np.vstack((X.ravel(), Y.ravel())).T

# Initialize an array to store the solution
u_fenics = np.zeros_like(points[:, 0])

# Evaluate the FEniCS solution at each point
for i, point in enumerate(points):
    u_fenics[i] = u(point)  # u is the solution obtained from FEniCS

# Reshape the solution to a 2D grid for plotting or comparison
u_fenics_grid = u_fenics.reshape((len(x), len(y)))

print("u_fenics", u_fenics)


c = plot(u)
plt.colorbar(c)
plt.title("Solution of Laplace's equation")
plt.xlabel('x')
plt.ylabel('y')
plt.savefig('solution_plot.png')  # Save the figure before showing
plt.show()

# Create a 3D plot
fig = plt.figure(figsize=(8, 6))
ax = fig.add_subplot(111, projection='3d')  # Create a 3D subplot

x = np.linspace(0, 1, 20)
y = np.linspace(0, 1, 20)
X, Y = np.meshgrid(x, y)
points = np.vstack((X.ravel(), Y.ravel())).T

# Initialize an array to store the solution
u_fenics = np.zeros_like(points[:, 0])

# Evaluate the FEniCS solution at each point
for i, point in enumerate(points):
    u_fenics[i] = u(point)

# Reshape the solution to a 2D grid for plotting or comparison
u_fenics_grid = u_fenics.reshape((len(x), len(y)))

# Plot the surface
surf = ax.plot_surface(X, Y, u_fenics_grid, cmap='viridis', edgecolor='k')
plt.colorbar(surf)
plt.title("3D plot of the solution")
plt.xlabel('x')
plt.ylabel('y')
ax.set_zlabel('u')
plt.show(block=True)

