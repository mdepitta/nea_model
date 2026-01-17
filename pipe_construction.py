import numpy as np
from pymadcad.geometry import extrude, revolve, shapes
from pymadcad.drawing import *
from pymadcad import Mesh, transform

# 1. Define the arbitrary midline (a path of points)
# This example creates an S-shaped curve
midline_points = np.array([
    [0.0, 0.0, 0.0],
    [1.0, 0.0, 0.0],
    [2.0, 1.0, 0.0],
    [2.0, 2.0, 0.0],
    [1.0, 3.0, 0.0],
    [0.0, 4.0, 0.0]
])

# Convert the numpy points into a pymadcad Wire object (path)
# A softened path can handle curves smoothly
path = Softened([point for point in midline_points])

# 2. Define the pipe profile (a 2D shape, e.g., a circle)
# This creates a circle with a radius of 0.2
pipe_radius = 0.2
profile_shape = shapes.circle(radius=pipe_radius, div=16) # div is the number of segments

# 3. Generate the tube mesh
# The 'tube' function sweeps the profile along the path
# section=True ensures the profile remains rigid and correctly oriented along the curve
pipe_mesh = tube(profile_shape, path, section=True)

# 4. (Optional) Visualize or export the mesh
# You can use pymadcad's draw function for visualization
draw(pipe_mesh, color="grey")

# To export as a common mesh format (e.g., STL, OBJ), you would typically
# need to use another library like `trimesh` or `meshio` to handle the file saving,
# as pymadcad focuses on generation.