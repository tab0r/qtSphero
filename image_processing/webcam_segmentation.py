import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import pygame.camera
import pygame.image
import time

from skimage import io, exposure
from skimage.filters import threshold_otsu
from skimage.segmentation import clear_border
from skimage.measure import label, regionprops
from skimage.morphology import closing, square
from skimage.color import label2rgb, rgb2gray
from skimage.transform import downscale_local_mean

# begin timing
t0 = time.time()

# # capture image from webcam
# pygame.camera.init()
# cam = pygame.camera.Camera(pygame.camera.list_cameras()[0])
# cam.start()
# img = cam.get_image()
# pygame.image.save(img, "photo.bmp")
# pygame.camera.quit()

# begin segmentation process
im_file = io.imread("photo.bmp")
# scaled = downscale_local_mean(im_file, (1, 1, 1))
# image = scaled[:, :, 2]
img = rgb2gray(im_file)
# io.imsave("images/webcam_test.png", image)
image = exposure.adjust_gamma(img, 2)
# Logarithmic
# logarithmic_corrected = exposure.adjust_log(img, 1)


t1 = time.time()
print("Image capture time: ", t1-t0)

# apply threshold
thresh = threshold_otsu(image)
bw = closing(image > thresh, square(5))

# remove artifacts connected to image border
cleared = clear_border(bw)

# label image regions
label_image = label(cleared)
t2 = time.time()
print("Segmentation time: ", t2 - t1)
image_label_overlay = label2rgb(label_image, image=image)

fig, ax = plt.subplots(figsize=(10, 6))
ax.imshow(image_label_overlay)

for region in regionprops(label_image):
    # take regions with large enough areas
    if region.area >= 20:
        # draw rectangle around segmented coins
        minr, minc, maxr, maxc = region.bbox
        rect = mpatches.Rectangle((minc, minr), maxc - minc, maxr - minr, fill=False, edgecolor='red', linewidth=2)
        ax.add_patch(rect)
        centroid = (minc + 0.5*(maxc - minc), minr + 0.5*(maxr - minr))
        circ = mpatches.Circle(centroid, radius = 5, fill=False, edgecolor='blue', linewidth=2)
        ax.add_patch(circ)
        print("Centroid of blob: ", centroid)
ax.set_axis_off()
plt.tight_layout()
plt.savefig("images/webcam_seg0.png")
t3 = time.time()
print("Save time: ", t3 - t2)
print("Total wall time: ", t3 - t0)
