from fenics import *
mesh = UnitSquareMesh(8, 8)
print("Mesh created successfully with %d cells" % mesh.num_cells())