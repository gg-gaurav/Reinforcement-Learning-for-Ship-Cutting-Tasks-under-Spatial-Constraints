import numpy as np
from scipy.spatial.transform import Rotation as R
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Original vector (surface normal)
v = np.array([0, 0, 1])

# Example quaternion (x, y, z, w) - rotate 90° around x-axis
q = R.from_euler('x', 90, degrees=True).as_quat()  # returns [x, y, z, w]

# Rotate vector
r = R.from_quat(q)
v_rot = r.apply(v)

# Plot
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# Plot original
ax.quiver(0, 0, 0, *v, color='blue', label='Original Normal')
# Plot rotated
ax.quiver(0, 0, 0, *v_rot, color='red', label='Rotated Normal')

ax.set_xlim([-1, 1])
ax.set_ylim([-1, 1])
ax.set_zlim([-1, 1])
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
ax.legend()
plt.title('Quaternion Rotation of a Surface Normal')
plt.show()
