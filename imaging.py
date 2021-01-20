import nibabel as nib
import numpy as np
import scipy.stats
import matplotlib.pyplot as plt
from tkinter import Tk
from tkinter.filedialog import askopenfilename
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import pandas as pd
from scipy.ndimage import label, find_objects, binary_closing, binary_opening, binary_dilation, binary_erosion
import progressbar, click


"""imaging.ipynb

Author: Stu Maitland

Original file is located at
    https://colab.research.google.com/drive/13xFHFDz54hQQEgPs1LAPaoYDi76gcyDq

# MUMRI Alternation Analysis
"""

"""Slice the data appropriately
Slices are arranged [x,y,slice,time]. 
We only want the second slice, this is where the MU data is. 
We can also trim the images to remove extra dark crud around them
"""

tk_ui = Tk()
tk_ui.withdraw()  # we don't want a full GUI, so keep the root window from appearing
filename = askopenfilename()  # show an "Open" dialog box and return the path to the selected file

tk_ui.destroy()
n1_img = nib.load(filename)
print('Filename: "{}" loaded.'.format(filename))

n1_data = n1_img.get_fdata()
stir_slice = n1_data[:, :, 1, :]

"""Firstly, measure the standard deviation over a time period considered 
'baseline' (the first 100 slices). Next, we measure the exponentially weighted
standard deviation, weighted towards the images at higher currents. This 
enables detection of alternation earlier, but in the context of the normal 
background variance seen e.g. due to brightness changes.
Divide the weighted standard deviation at each time point by the baseline to
produce a measure of comparative variability.
"""
repeat = True
prev_wlen = -1
while repeat:
    # Parameters:
    wlen = click.prompt('\nSelect averaging window length ', default=30)
    # Window length with which to average over consecutive images. Default = 30
    if wlen != prev_wlen:  # Skip this section if the user doesnt change the window length
        print('Measuring exponentially weighted intensity')
        ewstd = np.zeros((stir_slice.shape[0], stir_slice.shape[1], stir_slice.shape[2]))
        for x in progressbar.progressbar(range(1, stir_slice.shape[0])):
            for y in range(1, stir_slice.shape[1]):
                pand_stir = pd.Series(stir_slice[x, y, :])
                # Exponentially weighted standard deviation
                baseline = np.mean(pand_stir[10:60])
                ewstd[x, y, :] = pand_stir.ewm(halflife=5, ignore_na=True).mean() / baseline
    prev_wlen = wlen
    # Get the global signal- the mean of the signal intensity across all voxels at
    # each time point
    global_sig = np.sum(stir_slice, axis=(0, 1))

    """Next, we plot the average exponentially weighted standard deviation to check
     if the signal varies in the same sequences as we'd expect alternation"""

    vars = np.nanmean(np.nanmean(ewstd, axis=0), axis=0).flatten()
    plt.plot(vars)
    plt.xlabel('Image number')
    plt.ylabel('Relative variance above baseline')
    plt.suptitle('Change in intensity beyond normal baseline')
    plt.show()

    """ We set a per-voxel threshold for variance, above which we consider a voxel 
    to be alternating, in order to produce a binary mask of putative motor units"""

    # Parameters

    act_lim = click.prompt('\nSelect activation limit (0.3-0.7)', default=0.5)
    # !! Limit for alternation threshold adjusted between 0.3-0.7 Std devs above mean

    bin_mask = ewstd < act_lim
    counts = np.count_nonzero(np.count_nonzero(bin_mask, axis=0), axis=0)
    out = ewstd
    # ewstd.shape
    #plt.plot(counts)
    #plt.suptitle('Number of voxels meeting criteria')
    #plt.xlabel('Image number')
    #plt.show()

    """With this binary 3D image created, we can use morphological binary image 
    manipulation to remove noise and adjust thresholds for discrete objects
    
    **Morphological Opening**
    Removes small objects, used to hide noise from images e.g. an isolated variable
    voxel due to noise.
    
    **Morphological Closing**
    
    Removes small holes in larger objects (e.g. where the centre of an MU briefly 
    stops alternating)
    """

    t_dim = click.prompt('\nSelect closing z dimension', default=5)
    print('Closing...')
    closed = binary_closing(bin_mask, structure=np.ones((1, 1, t_dim)))
    # !! Tuning parameter- Adjust between (1,1,5) to (1,1,15)

    t_dim = click.prompt('\nSelect opening z dimension', default=12)
    print('Opening...')
    opened = binary_opening(closed, structure=np.ones((3, 3, t_dim)))
    # !! Tuning parameter- Adjust between (3,3,5) to (3,3,15)

    opened = binary_opening(opened, structure=np.ones((1, 1, 50)))
    # Do not tune
    regs, n_points = label(opened)
    print('{} Motor Units found'.format(n_points))
    objs = find_objects(regs)

    spat_size = []
    volumes = []
    corr_sig = []
    mu_map = np.zeros((stir_slice.shape[0], stir_slice.shape[1]))

    save_c= click.confirm('\nSave masks? ', default=True)
    print('Finding MUs:')
    for x in progressbar.progressbar(range(1, n_points)):
        dims = objs[x]
        roi = regs == x
        roi_sig = np.nansum(np.ma.masked_array(stir_slice, ~roi), axis=(0, 1))
        c = np.corrcoef(global_sig, roi_sig)[1, 0]
        corr_sig.append(c)
        map_2d = (np.sum(roi, axis=2) > 0) * x
        mu_map += map_2d
        if save_c:
            volumes.append(np.sum(np.sum(np.sum(roi, axis=0), axis=0), axis=0))
            spat_size.append(np.sqrt((dims[1].stop - dims[1].start) ^ 2 + (dims[0].stop - dims[0].start) ^ 2))
            mask = nib.Nifti1Image(roi.astype(int), n1_img.affine)
            direc = "/".join(filename.split('/')[:-1])
            direc += '/mask_{}.nii'.format(x)
            nib.save(mask, direc)
    masked = np.ma.masked_where(mu_map == 0, mu_map)
    plt.imshow(stir_slice[:, :, 1].T, origin="upper", cmap="gray")
    plt.imshow(masked.T, cmap="jet", origin="upper", alpha=0.9)
    plt.suptitle('MU maps')
    plt.show()
    repeat = click.confirm('\nRedo analysis?', default=False)

sorted_indices = np.argsort(corr_sig)[::-1]


def save_to_video(filename):
    """If we want to export the variances as a video, this is how we save it out."""
    frames = []  # for storing the generated images
    fig = plt.figure()
    for i in range(regs.shape[2]):
        ax1 = plt.imshow(regs[:, :, i].T, cmap="jet", origin="upper", animated=True)
        frames.append([ax1])

    ani = animation.ArtistAnimation(fig, frames, interval=50, blit=True,
                                    repeat_delay=1000)
    ani.save(filename)

