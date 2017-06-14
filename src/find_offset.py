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
from keras.models import Sequential, load_model
from keras.layers.core import Dense
from keras.optimizers import sgd, RMSprop, Adagrad, Adam

def dir_angle(state, goal = (20, 15)):
    dx = goal[0] - state[0]
    dy = goal[1] - state[1]
    if dy == 0: dy = 0.1
    dir_angle = np.arctan(dx/dy)
    if dy < 0: dir_angle += np.pi       
    return dir_angle


def observe_state(regions, image, filename = "last_frame"):
    '''
    IN: regions of an image, an image to display with it
    OUT: the position and goal of the Sphero
    '''
    fig, ax = plt.subplots(figsize=(8, 6))
    # ax = fig.gca()
    ax.imshow(image)

    goal = (20, 15)
    circ = mpatches.Circle(goal, radius = 1, fill=False, edgecolor='red', linewidth=2)
    ax.add_patch(circ)
    circ = mpatches.Circle(goal, radius = 2, fill=False, edgecolor='blue', linewidth=2)
    ax.add_patch(circ)

    centroids = []
    for region in regions:
            # draw rectangle around segmented regions
            minr, minc, maxr, maxc = region.bbox
            rect = mpatches.Rectangle((minc, minr), maxc - minc, maxr - minr, fill=False, edgecolor='red', linewidth=2)
            ax.add_patch(rect)
            centroid = (minc + 0.5*(maxc - minc), minr + 0.5*(maxr - minr))
            centroids.append(centroid)
            circ = mpatches.Circle(centroid, radius = 2, fill=False, edgecolor='blue', linewidth=2)
            ax.add_patch(circ)
            arrow = mpatches.FancyArrowPatch(goal, centroid, color = 'yellow')
            ax.add_patch(arrow)

    ax.set_axis_off()
    output = []
    if len(centroids) > 0:
        distance = dist_e(goal, centroids[0])
        state = list(centroids[0])
        # state.extend(centroids[0])
        rad_angle = dir_angle(state)
        dir_coords = (centroid[0] + 10*np.sin(rad_angle), 
                        centroid[1] + 10*np.cos(rad_angle))
        dir_arrow = mpatches.FancyArrowPatch(centroid, dir_coords, color = 'green')
        ax.add_patch(dir_arrow)
        if distance < 10.0:
            reward = (10.0 - distance) / 10
        else:
            reward = -0.01
    else:
        rad_angle = 0
        state = [0,0]
        distance = 0.00
        reward = 0.00
    output.append(state)
    output.append(reward)
    output.append(distance)
    dist_str = "Distance: " + "{0:.2f}".format(distance)
    # output.append(dist_str)
    reward_str = "Reward: " + "{0:.2f}".format(reward)
    angle_str = "Angle to target: " + "{0:.2f}".format(rad_angle)
    # output.append(reward_str)
    plt.title(reward_str)
    plt.title(dist_str, loc="left")
    plt.title(angle_str, loc="right")
    plt.tight_layout()
    # plt.show()
    plt.savefig(filename + ".png")
    plt.close()
    return output

def step_game(sphero, cam, e = 0.1, model = None, last_input = None, last_choice = None):
    _ = capture_image(cam)
    labelled_image, image, _ = segment_photo_bmp()
    filtered_regions = filter_regions(labelled_image)
    data = observe_state(filtered_regions, image)
    print("CV data: ", data)
    # special sauce goes here
    if model!=None:
        coord_est = data[0]
        # pdb.set_trace()
        camera_matrix = image
        # append those two here, feed it as input
        # or just use camera?
        pre_input = image
        inputs = np.array(pre_input).reshape(1, 1200)
        reward = data[1]
        predicts = model.predict(inputs)
        distance = data[2]
        loss = 0
        if (last_input != None) and (last_choice != None):
            # pdb.set_trace()
            target = np.zeros(4)
            target[last_choice] = reward + predicts[0][last_choice]
            loss += model.train_on_batch(last_input, np.array(target).reshape(1,4))
    else:
        inputs = data
        loss = 0
        predicts = "none"
        reward = "none"
        distance = "none"
        choice = randint(0, 4)
        predicts = np.zeros(5)
        predicts[choice] = 0.1
    # print("Inputs: ", inputs)
    print("Loss: ", loss)
    print("Predictions: ", predicts)#np.argmax(predicts))
    print("Reward: ", reward)
    print("Distance: ", distance)
    if sphero != None:
        # dir choice is the angle in "camera space" to our target
        dir_choice = dir_angle(coord_est) * float(180)/np.pi
        if np.random.rand() < e:
            # offset choice is the angle between the camera zero and the Sphero zero. This is what we try to learn.
            offset_choice = randint(0,4)
        else: 
            offset_choice = np.argmax(predicts)
        if dir_choice != None:
            # our final direction output is the deterministically chosen direction plus an offset choice times 360/n, where n is the number of choices we gave the network. Hopefully, this stochastic method will have some positive results for four choices, then extend to a larger number.
            direction = ((90 * offset_choice) + int(dir_choice)) % 360
            speed = 28
            sphero.roll(speed, direction)
        time.sleep(0.5)
    return inputs, predicts, reward, distance


def baseline_model(optimizer = Adam(),
                    layers = [{"size":80,"activation":"relu"}]):
    # two inputs - each coordinate
    inputs = 1200
    # four outputs - one for each potential offset
    num_outputs = 4
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
    model.add(Dense(num_outputs, activation='relu'))
    model.compile(optimizer = optimizer,
                    loss = "mean_squared_error")
    return model

# '68:86:E7:06:FD:1D', '68:86:E7:07:07:6B', '68:86:E7:08:0E:DF'
def pygame_play(cam, addr, n = 10, episodes = 10):
    pygame.display.init()
    screen = pygame.display.set_mode((800, 600))
    white = (255, 64, 64)
    if os.path.isfile('models/sphero_find_offset.h5') == True:
        model = load_model( "models/sphero_find_offset.h5")
        print("Saved model loaded")
    else:
        model = baseline_model()
    print("Bringing Sphero online")
    with Kulka(addr) as sphero:
        print("Sphero online!")
        # spheros.append(sphero)
        sphero.set_rgb(0, 0, 0x0F)
        _ = input("Press Ctrl-D to abort, or any other key to continue training")
        for _ in range(episodes):
            # for sphero in spheros:
            sphero.set_rgb(0xFF, 0xFF, 0xFF)
            # sphero.set_rgb(0, 0, 0)
            log = []
            sphero.set_inactivity_timeout(300)
            # cam = cam_setup(i = 1)
            log_i = step_game(sphero, cam, e = 0.2, model = model)
            img = pygame.image.load('last_frame.png')
            screen.fill((white))
            screen.blit(img,(0,0))
            pygame.display.flip()
            log.append(log_i)
            for i in range(n-1):
                log_i = step_game(sphero, cam, model = model, \
                    last_input = np.array(log[i][0][0]).reshape(1,1200), \
                    last_choice = np.argmax(log[i][1]))
                img = pygame.image.load('last_frame.png')
                screen.fill((white))
                screen.blit(img,(0,0))
                pygame.display.flip()
                log.append(log_i)
                time.sleep(0.5)
            sphero.set_rgb(0, 0, 0x0F)
        # for sphero in spheros:
    # sphero.close()
    # cam_quit(cam)
    overwrite = input("Overwrite old model? (y/n)")
    if overwrite == 'y':
        model.save("models/sphero_find_offset.h5" )
        print("Model updated")
    else:
        model.save("models/sphero_find_offset_p.h5" )
        print("Prime model saved")

def dry_run(cam, n = 10, episodes = 10):
    pygame.display.init()
    screen = pygame.display.set_mode((800, 600))
    white = (255, 64, 64)
    _ = input("Press Ctrl-D to abort, or any other key to continue training")
    for _ in range(episodes):
        # for sphero in spheros:
        # sphero.set_rgb(0xFF, 0xFF, 0xFF)
        # sphero.set_rgb(0, 0, 0)
        # log = []
        # sphero.set_inactivity_timeout(300)
        # cam = cam_setup(i = 1)
        model = None
        _ = step_game(cam = cam, sphero = None, e = 0.2, model = model)
        img = pygame.image.load('last_frame.png')
        screen.fill((white))
        screen.blit(img,(0,0))
        pygame.display.flip()
        for i in range(n-1):
            _ = step_game(cam = cam, sphero = None, model = model)
            img = pygame.image.load('last_frame.png')
            screen.fill((white))
            screen.blit(img,(0,0))
            pygame.display.flip()

def one_image(i = 1):
    cam = cam_setup(i)
    _ = capture_image(cam)
    labelled_image, image, _ = segment_photo_bmp()
    filtered_regions = filter_regions(labelled_image)
    observe_state(filtered_regions, image)
    cam_quit(cam)


if __name__ == "__main__":
    one_image()
    addrs = ['68:86:E7:06:FD:1D', '68:86:E7:07:07:6B', '68:86:E7:08:0E:DF']
    # cam = cam_setup(1)
