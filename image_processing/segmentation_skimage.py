import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

from skimage import data, io
from skimage.filters import threshold_otsu
from skimage.segmentation import clear_border
from skimage.measure import label, regionprops
from skimage.morphology import closing, square
from skimage.color import label2rgb, rgb2gray
from skimage.transform import downscale_local_mean

im_file = io.imread("images/sphero6.jpg")
#image = data.coins()[50:-50, 50:-50]
# image = scaled[:, :, 0]
im_gray = rgb2gray(im_file)
image = downscale_local_mean(im_gray, (10, 10))
io.imsave("images/sphero6.png", image)

# apply threshold
thresh = threshold_otsu(image)
bw = closing(image > thresh, square(5))

# remove artifacts connected to image border
cleared = clear_border(bw)

# label image regions
label_image = label(bw)
image_label_overlay = label2rgb(label_image, image=image)

fig, ax = plt.subplots(figsize=(10, 6))
ax.imshow(image)#_label_overlay)

for region in regionprops(label_image):
    # take regions with large enough areas
    if region.area >= 100:
        # draw rectangle around segmented coins
        minr, minc, maxr, maxc = region.bbox
        rect = mpatches.Rectangle((minc, minr), maxc - minc, maxr - minr, fill=False, edgecolor='red', linewidth=2)
        ax.add_patch(rect)

ax.set_axis_off()
plt.tight_layout()
plt.savefig("images/sphero_seg6.png")
