import os
import sys
import time
import pygame
import numpy as np

from webcam_segmentation import cam_setup, cam_quit, capture_image, region_centroids, filter_regions
from pygame.locals import *
from random import randint

from scipy.spatial.distance import euclidean as dist_e

from skimage import io, exposure
from skimage.filters import threshold_otsu
from skimage.segmentation import clear_border
from skimage.measure import label, regionprops
from skimage.morphology import closing, square
from skimage.color import label2rgb, rgb2gray, gray2rgb
from skimage.transform import downscale_local_mean
sys.path.append(os.path.abspath("../../kulka"))
from kulka import Kulka

def segment_surface(surface):
	'''
	IN: pygame surface, from pygame camera, for segmenting
	OUT: label_image, two arrays ready for surface-ing, and segmentation time
	'''
	# begin timing
	t0 = time.time()
	# begin segmentation process
	t1 = time.time()
	# image = scaled[:, :, 2]
	# surface = pygame.transform.flip(surface, True, False)
	rgb = pygame.surfarray.array3d(surface)
	img = rgb2gray(rgb)
	# scaled = downscale_local_mean(img, (16, 16))
	image = exposure.adjust_gamma(img, 10)
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
	bw_surf = (gray2rgb(bw)*255).astype('uint8')
	# the factor of 15 is because of how small values we get from the exposure adjustment. Probly will need to be tweaked 
	img_surf = (gray2rgb(image)*255*15).astype('uint8')
	return label_image, img_surf, bw_surf, t2 - t0

pygame.display.init()
screen = pygame.display.set_mode((640, 480))
BLACK = (  0,   0,   0)
WHITE = (255, 255, 255)
RED   = (255,   0,   0)
GREEN = (  0, 255,   0)
BLUE  = (  0,   0, 255)

screen.fill((WHITE))
cam = cam_setup(1)
frames = 4
t0 = time.time()
addrs = [
	'68:86:E7:06:FD:1D',
	'68:86:E7:07:07:6B',
	'68:86:E7:08:0E:DF']
    #
i = 1
with Kulka(addrs[i]) as kulka:
	kulka.set_inactivity_timeout(300)
# while True:
	for i in range(20):
		# direction = np.argmax(model.predict(last_frame))
		direction = randint(0, 359)
		kulka.roll(randint(25, 35), direction)
		frame = []
		distances = []
		rewards = []
		for f in range(frames):
			cam_img = capture_image(cam, None, False)
			label_img, gamma, bw, seg_time = segment_surface(cam_img)
			img = pygame.pixelcopy.make_surface(bw)
			regions = filter_regions(label_img, min_area = 5000, max_area = 12000)
				# , verbose = True)
			centroids = []
			for region in regions:
				# draw ciclre around objects
				minr, minc, maxr, maxc = region.bbox
				r = np.min([maxr-minr, maxc-minc])
				ctr = (minc + 0.5*(maxc - minc), minr + 0.5*(maxr - minr))
				centroids.append(ctr)
				pygame.draw.circle(img, GREEN, (int(ctr[1]), int(ctr[0])), 10, 5)
				pygame.draw.circle(img, RED, (int(ctr[1]), int(ctr[0])), r, 5)
			# while len(centroids) < 4:
			# 	centroids.append((0,0))
			# centroids = centroids[0:4]
			if len(centroids) > 0:
				centroid = centroids[0]
				centroid = (centroid[1], centroid[0])
			else:
				centroid = (0, 0)
			frame.append(centroid)
			pygame.draw.line(img, BLUE, centroid, (320, 240))
			distance = dist_e(centroid, (320, 240))
			distances.append(distance)
			tolerance = 200
			reward = 1 * (1 - ((distance)/tolerance))
			rewards.append(reward)
			pygame.draw.circle(img, BLUE, (320, 240), 15, 5)
			pygame.draw.circle(img, RED, (320, 240), 10, 5)
			screen.blit(img,(0,0))
			pygame.display.flip()
			time.sleep(3/frames)
			# print(len(centroids), centroids)
		print(np.array([frame]).flatten())
		print("mean distances: ", np.mean(distances))
		print("mean rewards: ", np.mean(rewards))
	# break
t1 = time.time()

print("time: ", t1-t0)
# time for 100 frames was 4.191658973693848, so 25 fps. makes sense
# time for segmenting 100 frames was about 16, so 13 fps.
# time for segmentation and centroid calculations on 100 frames was 20, so we still have 5 frames per second. we fucking golden!
# segmentation on 4 frames at a time, taking only one centroid per frame
# 100 frames took 96.67178440093994
# taking only 4 frames at a time took 60 seconds
# this could be perfect, the roll command takes about 1-3 seconds to execute. time to import the kulka stuff!