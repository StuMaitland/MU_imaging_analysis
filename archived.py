import numpy as np
import matplotlib.pyplot as plt

"""
# Unused parts

These are sections used during application development but not currently in use"""



def roi(slice, x, y, width):
    roi = slice[x:x + width, y:y + width, :]
    return roi


"""This section is to randomly or exhaustively measure the spearman correlation of all parts of the image, with time (a proxy for stimulation current)"""

t = np.arange(stir_slice.shape[2])
out = np.zeros((2, 1000))

for n in range(0, 1000):
    rnd_roi = select_random_roi(stir_slice, 5)
    u_inten = np.mean(rnd_roi, axis=(0, 1))
    corr = scipy.stats.spearmanr(u_inten, t)
    out[0, n] = corr.correlation
    out[1, n] = corr.pvalue
    # out.append([corr.correlation,corr.pvalue])
# print(out)
plt.scatter(out[0, :], out[1, :])
plt.xlabel('correlation coefficient')
plt.ylabel('p value')

dims = stir_slice.shape
print(dims)
width = 5

out = np.zeros((dims[0] - width, dims[1] - width))
pv = np.zeros((dims[0] - width, dims[1] - width))
base_vox = stir_slice[45, 98, :]
for x in range(1, dims[0] - width):
    for y in range(1, dims[1] - width):
        reg = roi(stir_slice, x, y, 5)
        u_inten = np.mean(reg, axis=(0, 1))
        corr = scipy.stats.spearmanr(base_vox, u_inten)
        out[x, y] = corr.correlation
        pv[x, y] = corr.pvalue

"""Display the resulting spearman correlation matrix for each voxel"""

fig, (ax1, ax2) = plt.subplots(figsize=(13, 3), ncols=2)
pos = ax1.imshow(out.T, cmap="jet", origin="upper")
fig.colorbar(pos, ax=ax1)
ax1.scatter(44, 97)

pos = ax2.imshow(pv.T < 0.01, cmap="jet", origin="upper")
fig.colorbar(pos, ax=ax2)

fig = plt.figure()
ax = fig.gca(projection='3d')
ax.voxels(out, edgecolor='k')

plt.show()