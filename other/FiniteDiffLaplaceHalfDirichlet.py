import numpy as np
import matplotlib.pyplot as plt

# Define the grid
n = 50  # number of grid points along x and y
x = np.linspace(0, 1, n)
y = np.linspace(0, 1, n)
dx = x[1] - x[0]
dy = y[1] - y[0]
X, Y = np.meshgrid(x, y)

# Initialize the solution array
u = np.zeros((n, n))

# Set the boundary conditions
# Upper boundary
u[-1, :n//2] = 0  # Zero Dirichlet for first half
u[-1, n//2:] = 1  # u=1 for second half

# Lower boundary
u[0, :n//2] = 1  # u=1 for first half
u[0, n//2:] = 0  # Zero Dirichlet for second half

# Zero Neumann conditions on the left and right boundaries will be automatically satisfied
# by not updating the values at those boundaries (i.e., they remain zero)

# Perform the numerical solution using finite differences
# Using Gauss-Seidel method
max_iter = 5000  # maximum number of iterations
tolerance = 1e-6  # convergence tolerance
for it in range(max_iter):
    u_old = u.copy()
    for i in range(1, n-1):
        for j in range(1, n-1):
            u[i, j] = 0.25 * (u[i+1, j] + u[i-1, j] + u[i, j+1] + u[i, j-1])
    # Check for convergence
    if np.linalg.norm(u - u_old, np.inf) < tolerance:
        break

# Plot the solution
plt.figure(figsize=(8, 8))
plt.contourf(X, Y, u, levels=50, cmap='viridis')
plt.colorbar(label='u(x, y)')
plt.xlabel('x')
plt.ylabel('y')
plt.title('Solution of Laplace Equation with Mixed Boundary Conditions')
plt.show()


from mpl_toolkits.mplot3d import Axes3D

# Create a 3D plot
fig = plt.figure(figsize=(10, 10))
ax = fig.add_subplot(111, projection='3d')

# Plot the surface
surface = ax.plot_surface(X, Y, u, cmap='viridis')
fig.colorbar(surface, ax=ax, label='u(x, y)')

# Labels and title
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_zlabel('u(x, y)')
ax.set_title('3D Plot of Laplace Equation Solution with Mixed Boundary Conditions')

plt.show()