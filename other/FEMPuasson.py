from fenics import *
import matplotlib.pyplot as plt
import numpy as np

# Create mesh and define function space
nx, ny = 100, 100  # number of nodes in the x and y directions
mesh = UnitSquareMesh(100, 100)
V = FunctionSpace(mesh, 'P', 1)

# Define boundary conditions
def boundary_upper(x, on_boundary):
    return on_boundary and near(x[1], 1.0)

def boundary_lower(x, on_boundary):
    return on_boundary and near(x[1], 0.0)

bc_upper = DirichletBC(V, Constant(0.0), boundary_upper)  # Zero Dirichlet condition on upper boundary
bc_lower = DirichletBC(V, Constant(0.0), boundary_lower)  # One Dirichlet condition on lower boundary

bcs = [bc_upper, bc_lower]

# Define problem
u = TrialFunction(V)
v = TestFunction(V)
f = Expression('-pi * sin(pi * x[0]) * sin(pi * x[1])', degree=2)
a = -dot(grad(u), grad(v))*dx
L = f*v*dx

# Compute solution
u = Function(V)
solve(a == L, u, bcs)

# Plot solution using matplotlib
c = plot(u)
plt.colorbar(c)
plt.title("FEM solution of Laplace's equation")
plt.xlabel('x')
plt.ylabel('y')
plt.savefig('fenics_solution_half_dirichlet_Laplace_heatmap.png')
plt.show()

# Create a 3D plot
fig = plt.figure(figsize=(8, 6))
ax = fig.add_subplot(111, projection='3d')

x = np.linspace(0, 1, 100)
y = np.linspace(0, 1, 100)
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