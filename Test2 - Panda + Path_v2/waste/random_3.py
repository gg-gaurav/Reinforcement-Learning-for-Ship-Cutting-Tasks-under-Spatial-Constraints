import pybullet as p
import numpy as np
import pybullet_data
draw = 1
printtext = 0

if (draw):
  p.connect(p.GUI)
else:
  p.connect(p.DIRECT)


def check(point, box_pos, box_orn, half_extents):
    inv_pos, inv_orn = p.invertTransform(box_pos, box_orn)
    local_pt, _= p.multiplyTransforms(
        positionA= inv_pos,
        orientationA= inv_orn,
        positionB= point,
        orientationB= np.array([0.0, 0.0, 0.0, 1.0])
    ) 
    flag = True
    for i in range(3):
        if local_pt[i] > -half_extents[i] and local_pt[i] < half_extents[i]:
            continue
        else:
            flag = False
    return flag



p.setAdditionalSearchPath(pybullet_data.getDataPath())

# Container center
container_center = np.zeros(3)

# HalfExtents
halfExtents = np.array([0.2, 0.1, 0.05]) 

# Orientation
orientation = np.array([0.0, 0.0, 0.0, 1.0])

rgba_color = np.array([0.1, 0.1, 0.9, 0.5])                      # colour for the sphere
basePosition = container_center
baseOrientation = orientation
visual_kwargs = {
    "halfExtents": halfExtents,
    "specularColor": None,
    "rgbaColor": rgba_color 
}
baseVisualShapeIndex = p.createVisualShape(shapeType= p.GEOM_BOX,
                                            **visual_kwargs)
baseCollisionShapeIndex = -1                                    # for ghost shapes CollisionShapeIndex = -1
container = p.createMultiBody(
    baseMass = 0.0,
    baseCollisionShapeIndex = baseCollisionShapeIndex,
    baseVisualShapeIndex = baseVisualShapeIndex,
    basePosition = basePosition,
    baseOrientation = baseOrientation
)

point = np.array([0,0.15,0])
radius = 0.03
rgba_color = np.array([0.1, 0.9, 0.1, 0.5])                     
basePosition = point

visual_kwargs = {
    "radius": radius,
    "specularColor": None,
    "rgbaColor": rgba_color}
baseVisualShapeIndex = p.createVisualShape(shapeType= p.GEOM_SPHERE,
                                                                        **visual_kwargs)
baseCollisionShapeIndex = -1                                    
target_sphere = p.createMultiBody(
            baseMass = 0.0,
            baseCollisionShapeIndex = baseCollisionShapeIndex,
            baseVisualShapeIndex = baseVisualShapeIndex,
            basePosition = basePosition)

print(f"check: {check(point, container_center, orientation, halfExtents)}")


while (1):
  a = 0
  p.stepSimulation()