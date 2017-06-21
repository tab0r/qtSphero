import os
import sys
import pdb
import time
import pygame
import pickle
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

# neural network imports
from keras.utils import np_utils
from keras.models import Sequential, load_model
from keras.layers.core import Dense
from keras.optimizers import Adam

# yay colors!
BLACK = (  0,   0,   0)
WHITE = (255, 255, 255)
RED   = (255,   0,   0)
GREEN = (  0, 255,   0)
BLUE  = (  0,   0, 255)

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

def observe(screen, cam, frames, o_time = 3, choice = 0):
	frame = []
	distances = []
	rewards = []
	for f in range(frames):
		cam_img = capture_image(cam, None, False)
		label_img, gamma, bw, seg_time = segment_surface(cam_img)
		img = pygame.pixelcopy.make_surface(bw)
		regions = filter_regions(label_img, min_area = 100, max_area = 12000)
			# , verbose = True)
		centroids = []
		for region in regions:
			# draw ciclre around objects
			minr, minc, maxr, maxc = region.bbox
			r = np.min([maxr-minr, maxc-minc])
			ctr = (minc + 0.5*(maxc - minc), minr + 0.5*(maxr - minr))
			centroids.append(ctr)
			pygame.draw.circle(img, GREEN, (int(ctr[1]), int(ctr[0])), 10, 5)
			pygame.draw.circle(img, RED, (int(ctr[1]), int(ctr[0])), np.max([8, r]), 5)
		# while len(centroids) < 4:
		# 	centroids.append((0,0))
		# centroids = centroids[0:4]
		if len(centroids) > 0:
			centroid = centroids[0]
			centroid = (centroid[1], centroid[0])
		else:
			centroid = (0, 0)
		frame.extend(list(centroid))
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
		time.sleep(o_time/frames)
		# print(len(centroids), centroids)	
	print("************************************************************")
	m_dist = np.mean(distances)
	print("mean distance: ", m_dist)
	reward = np.mean(rewards)
	print("mean reward: ", reward)
	frame.append(choice)
	frame = np.array(frame).flatten().reshape(1, 1 + 2*frames)
	print("frame:", frame)
	return frame, reward, distance

def parse_choice(choice):
	if choice == 8:
		return 0, 0
	else:
		if choice < 4:
			speed = 30
		else:
			speed = 45
		direction = (90 * choice) % 360
		return speed, direction

def game_episode(kulka, cam, frames = 5, steps = 50, e = 0.1, model = None, t_Lim = None):
	pygame.display.init()
	screen = pygame.display.set_mode((640, 480))
	screen.fill((WHITE))
	start_frame, _, _ = observe(screen, cam, frames)
	if (model == None) or (np.random.random() < e):
		predicts = np.array([np.random.random() for _ in range(9)]).reshape(1,9)
	else:
		predicts = model.predict(start_frame)
	total_loss = 0
	losses = []
	inputs = []
	targets = []
	last_frame = start_frame
	for _ in range(steps):
		choice = np.argmax(predicts)
		print("--------------Begin Step------------------")
		# print("Q-values: ", ['{0:.3g}'.format(p) for p in predicts[0]])
		print("Choice: ", choice)
		speed, direction = parse_choice(choice)
		if kulka != None:
			try:
				kulka.roll(speed, direction)
			except:
				if model != None:
					model.save("autosaved_model_x.h5")
					output = [inputs, targets, losses, total_loss]
				pickle.dump( output, open( "autosaved_data_x.p", "wb" ) )
				print("Connection lost, model saved if exists")
				# return log so far
				break
		frame, reward, distance = observe(screen=screen, cam=cam, frames=frames, o_time=t_Lim, choice=choice)
		if kulka != None:
			try:
				kulka.roll(0, direction)
			except:
				pass
		if (model == None) or (np.random.random() < e):
			new_predicts = np.array([np.random.random() for _ in range(9)]).reshape(1,9)
		else:
			new_predicts = model.predict(frame)
		# pdb.set_trace()
		target = predicts
		target[0][choice] = reward + 0.8*new_predicts[0][choice]
		# print("Q-value update: ", ['{0:.3g}'.format(p) for p in target[0]])
		target = np.array(target).reshape(1,9)
		# insert training code
		if model != None:
			# pdb.set_trace()
			loss = model.train_on_batch(last_frame, target)
			losses.append(loss)
			total_loss += loss
		inputs.append(last_frame)
		targets.append(target)
		# insert experience storage code
		predicts = new_predicts
		last_frame = frame
		print("-------------End Step------------------")
	return np.vstack(inputs), np.vstack(targets), np.array(losses), total_loss

def play_game(kulka = None, model = None, steps = 50, t_Lim = None):
	cam = cam_setup(1)
	t0 = time.time()
	output = game_episode(kulka, cam, steps = steps, model = model, e = 0.5, t_Lim = t_Lim)
	cam_quit(cam)
	t1 = time.time()

	print("game_time: ", t1-t0)
	return output

def baseline_model(optimizer = Adam(), inputs = 11, outputs = 9,
                    layers = [{"size":50,"activation":"relu"}]):
    # two inputs - each coordinate
    num_inputs = inputs
    # four outputs - one for each potential offset
    num_outputs = outputs
    # prepare the navigator model
    model = Sequential()
    # initial inputs
    l = list(layers)
    l0 = l[0]
    del l[0]
    model.add(Dense(l0['size'],
                    input_dim = num_inputs,
                    activation = l0['activation']))
    # the hidden layers
    for layer in l:
        model.add(Dense(layer['size'], activation=layer['activation']))
    # the output layer
    model.add(Dense(num_outputs, activation='tanh'))
    model.compile(optimizer = optimizer,
                    loss = "mean_squared_error")
    return model

if __name__ == "__main__":
	addrs = [
		'68:86:E7:06:FD:1D',
		'68:86:E7:07:07:6B',
		'68:86:E7:08:0E:DF']
	while True:
		i = int(input("Enter Sphero #: "))
		n = int(input("Enter number of steps: "))
		t = float(input("Enter a time limit for each step: "))
		epi = int(input("Enter number of episodes: "))
		model = baseline_model()
		with Kulka(addrs[i]) as kulka:
			for _ in range(epi):
				if os.path.isfile("sphero_model.h5") == True:
					print("Loading existing model")
					model.load_weights("sphero_model.h5") 
				output = play_game(kulka = kulka, model = model, steps = n, t_Lim = t)
				i = 0
				model.save("sphero_model.h5")
				model_filestr = "models/autosaved_model_" + str(i) + ".h5"
				data_filestr = "episodes/autosaved_data_" + str(i) + ".p"
				while(os.path.isfile(model_filestr) == True):
					i += 1
					model_filestr = "models/autosaved_model_" + str(i) + ".h5"
					data_filestr = "episodes/autosaved_data_" + str(i) + ".p"
				if model != None:
					replay_loss = model.train_on_batch(output[0], output[1])
					model.save(model_filestr)
				pickle.dump(output, open(data_filestr, "wb" ))


# time for 100 frames was 4.191658973693848, so 25 fps. makes sense
# time for segmenting 100 frames was about 16, so 13 fps.
# time for segmentation and centroid calculations on 100 frames was 20, so we still have 5 frames per second. we fucking golden!
# segmentation on 4 frames at a time, taking only one centroid per frame
# 100 frames took 96.67178440093994
# taking only 4 frames at a time took 60 seconds
# this could be perfect, the roll command takes about 1-3 seconds to execute. time to import the kulka stuff!