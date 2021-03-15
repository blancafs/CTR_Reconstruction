import numpy as np

world_coors = np.array([-81.7572, -108.6253, -168.9161])
hom_world_coors = np.array([-81.7572, -108.6253, -168.9161, 1])
print('homogenous world coors:', hom_world_coors, ", with shape:", hom_world_coors.shape, "\n")

c2_rotation = np.transpose(np.array([[0.0218, -0.5367, 0.8435],
                                     [0.9981, 0.0607, 0.0129],
                                     [-0.0581, 0.8416, 0.5370]]))

c2_translation = np.transpose(np.array([[14.570], [-319.7436], [288.8639]]))

c2_extrinsic = np.append(c2_rotation,  c2_translation, axis=1)

aug = np.array((0, 0, 0, 1))

c2_extrinsic = np.insert(c2_extrinsic, 3, aug, 0)

print('c2_extrinsic matrix:', c2_extrinsic, ", with shape:", c2_extrinsic.shape, "\n")

c2_intrinsic = np.array([[816.5455, 0, 0], [0, 816.7107, 0], [324.5504, 237.2101, 1]])
c2_intrinsic = np.transpose(c2_intrinsic)
c2_intrinsic = np.c_[c2_intrinsic, np.zeros(3)]
print('c2_intrinsic matrix:', c2_intrinsic, ", with shape:", c2_intrinsic.shape, "\n")

i2_coors = [68.1485, 35.2743]

# transformation
M = np.matmul(c2_intrinsic, c2_extrinsic)
print('M = K [R T]:', M, ", with shape:", M.shape, "\n")

imagecoors = np.matmul(M, hom_world_coors)
print('M W:', imagecoors, ", with shape:", imagecoors.shape, "\n")

# apply homogeneous normalisation
i2_x = imagecoors[0] / imagecoors[2]
i2_y = imagecoors[1] / imagecoors[2]
norm_i2coors = [i2_x, i2_y]

print('Value should be:', i2_coors, ", coors obtained are:", norm_i2coors)
