import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import pygame.camera
import pygame.image
import time
import pdb

from skimage import io, exposure
from skimage.filters import threshold_otsu
from skimage.segmentation import clear_border
from skimage.measure import label, regionprops
from skimage.morphology import closing, square
from skimage.color import label2rgb, rgb2gray
from skimage.transform import downscale_local_mean

def cam_setup(i = 0):
    pygame.camera.init()
    if i == 0:
        cam = pygame.camera.Camera(pygame.camera.list_cameras()[0])
    else:
        cam = pygame.camera.Camera(pygame.camera.list_cameras()[1],\
                                            (640, 480), "RGB")
    cam.start()
    return cam

def cam_quit(cam):
    cam.stop()
    pygame.camera.quit()

def capture_image(cam, filestr = None):
    t0 = time.time()
    # # capture image from webcam
    # pygame.camera.init()
    # cam = pygame.camera.Camera(pygame.camera.list_cameras()[0])
    # cam.start()
    img = cam.get_image()
    pdb.set_trace()
    if filestr != None:
        name = filestr + ".bmp"
    else:
        name = "photo.bmp" 
    pygame.image.save(img, name)

    t1 = time.time()
    capture_time = t1-t0
    # print("Image capture time: ", capture_time)

    return capture_time

def segment_photo_bmp():
    # begin timing
    t0 = time.time()
    # begin segmentation process
    # make sure there's an image to segment
    try:
        im_file = io.imread("photo.bmp")
    except:
        capture_image()
    t1 = time.time()
    # image = scaled[:, :, 2]
    img = rgb2gray(im_file)
    scaled = downscale_local_mean(img, (16, 16))
    image = exposure.adjust_gamma(scaled, 10)
    # Logarithmic
    # image = exposure.adjust_log(scaled, 5)
    # image = scaled

    # io.imsave("webcam_test.png", image)
    # apply threshold
    thresh = threshold_otsu(image)
    bw = closing(image > thresh, square(1))

    # remove artifacts connected to image border
    # cleared = clear_border(bw)

    # label image regions
    label_image = label(bw)
    t2 = time.time()
    seg_time = t2 - t1
    # print("Segmentation time: ", t2 - t1)

    return label_image, scaled, t2 - t0

def region_centroids(labelled_image, min_area = 20):
    centroids = []
    for region in regionprops(labelled_image):
        if region.area > min_area:
            # calculate centroid
            minr, minc, maxr, maxc = region.bbox
            centroid = (minc + 0.5*(maxc - minc), minr + 0.5*(maxr - minr))
            print("Centroid of blob: ", centroid)
            centroids.append(centroid)
    return centroids

def filter_regions(labelled_image, min_area = 1, max_area = 150):
    filtered_labels = []
    for region in regionprops(labelled_image):
        # take regions with large enough areas
        if (region.area >= min_area) and (region.area < max_area):
            filtered_labels.append(region)
    return filtered_labels

def save_segmented_image(regions, image, filename = "test"):
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.imshow(image)

    for region in regions:
            # draw rectangle around segmented coins
            minr, minc, maxr, maxc = region.bbox
            rect = mpatches.Rectangle((minc, minr), maxc - minc, maxr - minr, fill=False, edgecolor='red', linewidth=2)
            ax.add_patch(rect)
            centroid = (minc + 0.5*(maxc - minc), minr + 0.5*(maxr - minr))
            circ = mpatches.Circle(centroid, radius = 5, fill=False, edgecolor='blue', linewidth=2)
            ax.add_patch(circ)

    ax.set_axis_off()
    plt.tight_layout()
    plt.savefig("seg_"+filename+".png")

def show_segmented_image(regions, image, filename = "test"):
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.imshow(image)

    for region in regions:
            # draw rectangle around segmented coins
            minr, minc, maxr, maxc = region.bbox
            rect = mpatches.Rectangle((minc, minr), maxc - minc, maxr - minr, fill=False, edgecolor='red', linewidth=2)
            ax.add_patch(rect)
            centroid = (minc + 0.5*(maxc - minc), minr + 0.5*(maxr - minr))
            circ = mpatches.Circle(centroid, radius = 3, fill=False, edgecolor='blue', linewidth=2)
            ax.add_patch(circ)

    ax.set_axis_off()
    plt.tight_layout()
    plt.show()

def main(n = 3, i = 0):
    cam = cam_setup(i)
    for i in range(n):
        _ = capture_image(cam)
        labelled_image, image, _ = segment_photo_bmp()
        filtered_regions = filter_regions(labelled_image)
        # image_label_overlay = label2rgb(labelled_image, image=image)
        show_segmented_image(filtered_regions, image)
        centroids = region_centroids(labelled_image)
    cam_quit(cam)

if __name__ == "__main__":
    main(1, i = 1)
