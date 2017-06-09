# in this, we attempt to train a Sphero to stay in the center of our view.
import os, sys, matplotlib, time, pdb, pickle
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

import matplotlib.backends.backend_agg as agg
from pygame.locals import *

from IPython import display
from scipy.spatial.distance import euclidean as dist_e
from random import randint
from webcam_segmentation import *

# make sure to clone my fork of the kulka library
sys.path.append(os.path.abspath("../../kulka"))
from kulka import Kulka

# neural network imports
from keras.utils import np_utils
from keras.models import Sequential
from keras.layers.core import Dense
from keras.optimizers import sgd, RMSprop, Adagrad, Adam

def observe_state(regions, image, filename = "last_frame"):
    '''
    IN: regions of an image, an image to display with it
    OUT: the position and goal of the Sphero
    '''
    fig, ax = plt.subplots(figsize=(8, 6))
    # ax = fig.gca()
    ax.imshow(image)

    goal = (320, 240)
    circ = mpatches.Circle(goal, radius = 5, fill=False, edgecolor='red', linewidth=2)
    ax.add_patch(circ)
    circ = mpatches.Circle(goal, radius = 10, fill=False, edgecolor='blue', linewidth=2)
    ax.add_patch(circ)

    centroids = []
    for region in regions:
            # draw rectangle around segmented regions
            minr, minc, maxr, maxc = region.bbox
            rect = mpatches.Rectangle((minc, minr), maxc - minc, maxr - minr, fill=False, edgecolor='red', linewidth=2)
            ax.add_patch(rect)
            centroid = (minc + 0.5*(maxc - minc), minr + 0.5*(maxr - minr))
            centroids.append(centroid)
            circ = mpatches.Circle(centroid, radius = 5, fill=False, edgecolor='blue', linewidth=2)
            ax.add_patch(circ)
            arrow = mpatches.FancyArrowPatch(goal, centroid, color = 'yellow')
            ax.add_patch(arrow)

    ax.set_axis_off()
    output = []
    if len(centroids) > 0:
        distance = dist_e(goal, centroids[0])
        state = list(centroids[0])
        # state.extend(centroids[0])
        if distance < 100.0:
            reward = (100.0 - distance) / 100
        else:
            reward = -0.01
    else:
        state = [0,0]
        distance = 0.00
        reward = 0.00
    output.append(state)
    output.append(reward)
    output.append(distance)
    dist_str = "Distance: " + "{0:.2f}".format(distance)
    # output.append(dist_str)
    reward_str = "Reward: " + "{0:.2f}".format(reward)
    # output.append(reward_str)
    plt.title(reward_str)
    plt.title(dist_str, loc="left")
    plt.tight_layout()
    # plt.show()
    plt.savefig(filename + ".png")
    plt.close()
    return output

def step_game(sphero, cam, model = None, last_input = None, last_choice = None):
    _ = capture_image(cam)
    labelled_image, image, _ = segment_photo_bmp()
    filtered_regions = filter_regions(labelled_image, min_area = 500)
    data = observe_state(filtered_regions, image)
    print("CV data: ", data)
    # special sauce goes here
    if model!=None and data[0]!=[0,0]:
        inputs = np.array(data[0]).reshape(1, 2)
        reward = data[1]
        predicts = model.predict(inputs)
        distance = data[2]
        if (last_input != None) and (last_choice != None):
            # pdb.set_trace()
            target = np.zeros(5)
            target[last_choice] = reward + predicts[0][last_choice]
            model.train_on_batch(last_input, np.array(target).reshape(1,5))
    else:
        inputs = data
        predicts = "none"
        reward = "none"
        distance = "none"
        choice = randint(0, 4)
        predicts = np.zeros(5)
        predicts[choice] = 1
    print("Inputs: ", inputs)
    print("Q-predicts: ", predicts)
    print("Reward: ", reward)
    print("Distance: ", distance)
    if sphero != None:
        choice = np.argmax(predicts)
        if choice != 4:
            direction = choice * 90
            speed = 30
            sphero.roll(speed, direction)
    return inputs, predicts, reward, distance

def baseline_model(optimizer = Adam(),
                    layers = [{"size":20,"activation":"relu"}]):
    # two inputs - each coordinate
    inputs = 2
    # five outputs - one for each action
    # going with a square of unit movements since this is
    # discritized by our structure
    num_outputs = 5
    # prepare the navigator model
    model = Sequential()
    # initial inputs
    l = list(layers)
    l0 = l[0]
    del l[0]
    model.add(Dense(l0['size'],
                    input_dim = inputs,
                    activation = l0['activation']))
    # the hidden layers
    for layer in l:
        model.add(Dense(layer['size'], activation=layer['activation']))
    # the output layer
    model.add(Dense(num_outputs, activation='linear'))
    model.compile(optimizer = optimizer,
                    loss = "mean_squared_error")
    return model

#'68:86:E7:06:FD:1D',
def pygame_play(n = 10, addrs = ['68:86:E7:07:07:6B', '68:86:E7:08:0E:DF']):
    pygame.display.init()
    screen = pygame.display.set_mode((800, 600))
    white = (255, 64, 64)
    if os.path.isfile('/models/sphero_0.pkl') == True:
        model = pickle.load(open( "/models/sphero_0.pkl", "rb" ))
    else:
        model = baseline_model()
    for addr in addrs:
        print("Bringing Sphero online")
        with Kulka(addr) as sphero:
            print("Sphero online!")
            sphero.set_rgb(0xFF, 0xFF, 0xFF)
            # sphero.set_rgb(0, 0, 0)
            log = []
            sphero.set_inactivity_timeout(300)
            cam = cam_setup(i = 1)
            log_i = step_game(sphero, cam, model = model)
            img = pygame.image.load('last_frame.png')
            screen.fill((white))
            screen.blit(img,(0,0))
            pygame.display.flip()
            log.append(log_i)
            for i in range(n-1):
                log_i = step_game(sphero, cam, model = model, \
                    last_input = np.array(log[i][0][0]).reshape(1,2), \
                    last_choice = np.argmax(log[i][1]))
                img = pygame.image.load('last_frame.png')
                screen.fill((white))
                screen.blit(img,(0,0))
                pygame.display.flip()
                log.append(log_i)
        sphero.close()
        cam_quit(cam)
        pickle.dump( model, open( "models/sphero_0.pkl", "wb" ) )

def one_image(i = 1):
    cam = cam_setup(i)
    _ = capture_image(cam)
    labelled_image, image, _ = segment_photo_bmp()
    filtered_regions = filter_regions(labelled_image, min_area = 500)
    observe_state(filtered_regions, image)
    cam_quit(cam)

if __name__ == "__main__":
    one_image()
