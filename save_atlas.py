# Get DESIKAN-KILLIANY atlas from abagen and extract left and right surface
import numpy as np
import nibabel as nib
import abagen

atlas_3d = abagen.fetch_desikan_killiany()

print(atlas_3d['image'])
print(atlas_3d['info'])

# surface version
atlas = abagen.fetch_desikan_killiany(surface=True)
print(atlas['image'])

# left hemi
left = atlas['image'][0]
left_n = nib.load(left)
left_np = np.array(left_n.darrays[0].data)
print("\nLeft hemi:\n", left_np.shape, "\n", np.unique(left_np))

# right hemi
right = atlas['image'][1]
right_n = nib.load(right)
right_np = np.array(right_n.darrays[0].data)
print("\nRight hemi:\n", right_np.shape, "\n", np.unique(right_np))

# save surfaces in numpy arrays
np.save('data/left_desikan_surface.npy', left_np)
np.save('data/right_desikan_surface.npy', right_np)
