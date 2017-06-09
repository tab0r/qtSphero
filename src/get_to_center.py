# in this, we attempt to train a Sphero to stay in the center of our view.
import os
import sys
import time
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

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

def get_cv(regions, image, filename = "test"):
    '''
    IN: regions of an image, an image to display with it
    OUT: the position and goal of the Sphero
    '''
    fig, ax = plt.subplots(figsize=(10, 6))
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
        output.append(centroids[0])
        output.append(goal)
        if distance < 100.0:
            reward = (100.0 - distance) / 100
        else:
            reward = -0.01
    else:
        distance = 0.00
        reward = 0.00
    output.append(reward)
    dist_str = "Distance: " + "{0:.2f}".format(distance)
    reward_str = "Reward: " + "{0:.2f}".format(reward)
    plt.title(reward_str)
    plt.title(dist_str, loc="left")
    plt.tight_layout()
    # plt.show()
    return plt, output

def plot_cv(regions, image, filename = "test"):
    plt, out = get_cv(regions, image, filename)
    plt.show()
    return out

def step_game(sphero, cam, model=None, notebook = False):
    # kulka.set_rgb(0, 0, 0xFF)
    # time.sleep(0.1)
    _ = capture_image(cam)
    labelled_image, image, _ = segment_photo_bmp()
    filtered_regions = filter_regions(labelled_image, min_area = 500)
    if notebook == True:
        data = plot_cv(filtered_regions, image)
    else:
        plt, data = get_cv(filter_regions, image)
    if len(data) == 2:
        # special sauce goes here
        if model!=None:
            inputs = data
            predicts = model.predict(data)
            # direction = out[0]*360
            # speed = out[1]*60
        else:
            direction = randint(0, 359)
            speed = randint(0, 45)
            inputs.append("random")
    #     kulka.roll(speed, direction)
    # kulka.set_rgb(0, 0, 0)
    # time.sleep(0.1)
    # kulka.set_rgb(0xFF, 0, 0)
    if notebook == False:
        return inputs, predicts, plt

def notebook_game(cam, sphero = None, n = 5):
    # with Kulka(addrs[0]) as kulka:
    log = []
    # kulka.set_inactivity_timeout(300)
    cam = cam_setup()
    model = baseline_model()
    # inputs number of rounds
    for i in range(n):
        display.clear_output(wait=True);
        display.display(step_game(sphero, cam, notebook = True));
    # kulka.close()
    cam_quit(cam)

def baseline_model(optimizer = Adam(),
                    layers = [{"size":20,"activation":"relu"}]):
    # four inputs - each coordinate
    inputs = 4
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

def play_game(addrs = ['68:86:E7:06:FD:1D']):
    # with Kulka(addrs[0]) as kulka:
    sphero = None
    log = []
    # kulka.set_inactivity_timeout(300)
    cam = cam_setup()
    model = baseline_model()
    # inputs number of rounds
    for i in range(5):
        # input turns for this round
        log = step_game(sphero, cam, model = model)
        log.append([log[0], log[1]])
    # kulka.close()
    cam_quit(cam)
    return log

if __name__ == "__main__":
    cam = cam_setup()
    _ = capture_image(cam)
    labelled_image, image, _ = segment_photo_bmp()
    filtered_regions = filter_regions(labelled_image, min_area = 500)
    plot_cv(filtered_regions, image)
    cam_quit(cam)
