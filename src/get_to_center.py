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

# neural network imports go here

def first_test():
    addr = '68:86:E7:06:FD:1D'
    with Kulka(addr) as kulka:
        print("Sphero online, starting camera")
        cam = cam_setup()
        kulka.set_inactivity_timeout(300)
        for i in range(20):
            display.clear_output(wait=True);
            kulka.set_rgb(0xFF, 0, 0)
            # kulka.read_locator()
            # data = parse_locator(kulka.data_poll())
            # print(data)
            # coords = transform_sphero_coords((data['xpos'], data['ypos']))
            # print("Approx sphero coords: ", coords)
            capture_image(cam)
            regions, image, _ = segment_photo_bmp()
            filtered_regions = filter_regions(labelled_image, min_area = 500)
            centroids = region_centroids(filtered_regions)
            direction = randint(0, 359)
            speed = randint(0, 45)
            kulka.roll(speed, direction)
            display.display(show_cv(filtered_regions, image));
        cam_quit()
        kulka.close()

def parse_locator(data):
    data_list = []
    for kbyte in data[2]:
        data_list.append(kbyte)
        # print(type(kbyte))
    output = dict()
    output['xpos'] = 256*data_list[0] + data_list[1]
    output['ypos'] = 256*data_list[2] + data_list[3]
    output['xvel'] = 256*data_list[4] + data_list[5]
    output['yvel'] = 256*data_list[6] + data_list[7]
    for i, k in enumerate(output):
        # print(k, output[k])
        if output[k] > 32767:
            output[k] -= 65536
    output['sog'] = data_list[8] + data_list[9]
    return output

def transform_sphero_coords(coords):
    '''
    Take sphero locator coords and transform them to
    something useful for comparison to pixel values.

    Our .bmps are 640x480, the Sphero returns coordinates in
    range ~(-32k, 32k). That's a range of about 64,000, so we can just divide each coordinate by 100 to get the scale there. Then add 320 to take care of negative values.

    Consider cropping images to squares to improve fit.
    '''
    t_of_c = (coords[0]+320, coords[1]+240)
    return t_of_c
    # return coords

def show_cv(regions, image, filename = "test"):
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
    dist_str = "Distance: " + "{0:.2f}".format(distance)
    reward_str = "Reward: " + "{0:.2f}".format(reward)
    plt.title(reward_str)
    plt.title(dist_str, loc="left")
    plt.tight_layout()
    plt.show()
    return output

def notebook_cv(n = 10):
    cam = cam_setup()
    for i in range(n):
        display.clear_output(wait=True);
        _ = capture_image(cam)
        labelled_image, image, _ = segment_photo_bmp()
        filtered_regions = filter_regions(labelled_image, min_area = 500)
        display.display(show_cv(filtered_regions, image));
    cam_quit(cam)

def step_game(kulka, cam, n=5):
    kulka.set_rgb(0, 0, 0xFF)
    time.sleep(0.1)
    _ = capture_image(cam)
    labelled_image, image, _ = segment_photo_bmp()
    filtered_regions = filter_regions(labelled_image, min_area = 500)
    data = show_cv(filtered_regions, image)
    if len(data) == 2:
        # special sauce goes here
        direction = randint(0, 359)
        speed = randint(0, 45)
        kulka.roll(speed, direction)
    kulka.set_rgb(0, 0, 0)
    time.sleep(0.1)
    kulka.set_rgb(0xFF, 0, 0)

def setup_game(addrs = ['68:86:E7:06:FD:1D']):
    with Kulka(addrs[0]) as kulka:
        kulka.set_inactivity_timeout(300)
        cam = cam_setup()
        # inputs number of rounds
        for i in range(5):
            # input turns for this round
            play_game(kulka, n = 5)
        kulka.close()
        cam_quit(cam)

def play_game(sphero, cam, n = 5):
        for i in range(n):
            display.clear_output(wait=True);
            display.display(step_game(sphero, cam));

if __name__ == "__main__":
    cam = cam_setup()
    _ = capture_image(cam)
    labelled_image, image, _ = segment_photo_bmp()
    filtered_regions = filter_regions(labelled_image, min_area = 500)
    show_cv(filtered_regions, image)
    cam_quit(cam)
