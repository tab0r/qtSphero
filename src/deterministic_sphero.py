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

# this is a fixed speed for now
global_speed = 25

def angle_from_x(pos1, pos2 = (20, 15)):
    ''' returns an angle in radians between the x-axis and a line defined by two poitnts. '''
    dy = pos2[0] - pos1[0]
    dx = pos2[1] - pos1[1]
    if dy == 0: dy = 0.1
    # arctan2 returns values in the 
    dir_angle = (np.arctan2(dx, dy) + 3 * np.pi/2) % 2*np.pi
    return dir_angle

def find_angle_offset(sphero, cam, depth = 0):
    '''Observe state, get screen coords (x1, y1), roll along zero, get (x2, y2). Calculate screen angle and return in radians'''
    offsets = []
    while len(offsets) < 100:
        i = np.random.choice([0, 45, 90, 135, 180, 225, 270, 315])
        sphero.roll(0, 0)
        _ = capture_image(cam)
        labelled_image, image, _ = segment_photo_bmp()
        filtered_regions = filter_regions(labelled_image)
        coords_i = observe_state(filtered_regions)[0]
        sphero.set_rgb(0xFF, 0, 0)
        time.sleep(1)
        sphero.roll(global_speed, i)

        time.sleep(5)
        sphero.roll(0, 0)
        _ = capture_image(cam)
        labelled_image, image, _ = segment_photo_bmp()
        filtered_regions = filter_regions(labelled_image)
        coords_o = observe_state(filtered_regions)[0]
        if (coords_o == [0,0]) or (coords_i == [0,0]):
            pass
        else:
            dx = coords_o[1] - coords_i[1]
            dy = np.max([0.01, coords_o[0] - coords_i[0]])
            dist = dx*dx + dy*dy
            angle = int((np.arctan2(dy, dx)+np.pi) * 180/np.pi)
            # weight the offsets by distance travelled when calculated
            for i in range(int(dist)):
                offsets.append(angle)
            print(int(dist), angle)

    sphero.set_rgb(0xFF, 0xFF, 0xFF)
    avg = np.mean(offsets)
    print("Weighted mean offset: ", avg)
    return avg

def observe_state(regions):
    '''
    IN: regions of an image
    OUT: the position and goal of the Sphero
    '''
    goal = (20, 15)
    
    centroids = []
    for region in regions[:3]:
            # draw rectangle around segmented regions
            minr, minc, maxr, maxc = region.bbox
            centroid = (minc + 0.5*(maxc - minc), minr + 0.5*(maxr - minr))
            centroids.append(centroid)
    
    output = []
    if len(centroids) > 0:
        distance = dist_e(goal, centroids[0])
        state = list(centroids[0])
        # state.extend(centroids[0])
        rad_angle = angle_from_x(state)
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
    output.append(rad_angle)
    return output

def observe_and_show_state(regions, image, screen):
    # mode = "display", filename = "last_frame"
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
    for region in regions[:3]:
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
        rad_angle = angle_from_x(state)
        goal_coords = (centroid[0] + 5*np.cos(rad_angle), 
                        centroid[1] + 5*np.sin(rad_angle))
        goal_arrow = mpatches.FancyArrowPatch(centroid, goal_coords, color = 'green')
        ax.add_patch(goal_arrow)
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
    output.append(rad_angle)
    dist_str = "Distance: " + "{0:.2f}".format(distance)
    # output.append(dist_str)
    reward_str = "Reward: " + "{0:.2f}".format(reward)
    angle_str = "True angle: " + "{0:.2f}".format(rad_angle * float(180)/np.pi)

    # output.append(reward_str)
    plt.title(reward_str)
    plt.title(dist_str, loc="left")
    plt.title(angle_str, loc="right")
    plt.tight_layout()
    # plt.show()
    #plt.savefig(filename + ".png")

    canvas = agg.FigureCanvasAgg(fig)
    canvas.draw()
    renderer = canvas.get_renderer()
    raw_data = renderer.tostring_rgb()    
    plt.close()

    size = canvas.get_width_height()
    surf = pygame.image.fromstring(raw_data, size, "RGB")
    screen.blit(surf, (0,0))
    pygame.display.flip()

    return output

def step_det(sphero, cam, angle_offset = 0, display = True, screen = None):
    _ = capture_image(cam)
    labelled_image, image, _ = segment_photo_bmp()
    filtered_regions = filter_regions(labelled_image)
    if display == False:
        data = observe_state(filtered_regions)
    else:
        data = observe_and_show_state(filtered_regions, image, screen)
    coord_est = data[0]
    distance = dist_e(data[0], (20, 15))
    # dir choice is the angle in "camera space" to our target
    dir_choice = angle_from_x(coord_est)
    direction = angle_offset + int((dir_choice)*float(180)/np.pi) % 360
    if sphero != None:
        if distance > 5:
            sphero.set_rgb(0xFF, 0xFF, 0xFF)
            # speed = global_speed
            sphero.roll(global_speed, direction)
        else:
            direction = None
            sphero.set_rgb(0, 0xFF, 0)
            sphero.roll(0, 0)
        time.sleep(1)
        sphero.roll(0, 0)
    return coord_est, dir_choice, direction

# '68:86:E7:06:FD:1D', '68:86:E7:07:07:6B', '68:86:E7:08:0E:DF'
def play(cam, addr, n = 10, episodes = 10, mode = 0, display = True):
    if display == True:
        pygame.display.init()
        screen = pygame.display.set_mode((800, 600))
        white = (255, 64, 64)
    else:
        print("Beginning deterministic strategy tests")
        model = None
    print("Bringing Sphero online")
    with Kulka(addr) as sphero:
        print("Sphero online!")
        # spheros.append(sphero)
        sphero.set_inactivity_timeout(300)
        sphero.set_rgb(0, 0, 0x0F)
        # _ = input("Press Ctrl-D to abort, or enter to continue\n")
        angle_offset = 0 #randint(0, 359)
        for epi in range(episodes):
            print("Beginning episode ", epi)
            # for sphero in spheros:
            sphero.set_rgb(0xFF, 0xFF, 0xFF)
            # pdb.set_trace()
            print("Current offset angle: ", angle_offset)
            offset_a = input("Manually enter offset? (y/n)")
            if offset_a == 'y':
                new_offset = int(input("Enter a value in degrees on (0, 360)"))
                new_offset = (new_offset - 180)
            else:
                new_offset = find_angle_offset(sphero, cam)
            diff = new_offset - angle_offset
            if diff > 0.1:#/(1+epi): 
                print("Updating offset to ", new_offset)
                angle_offset = new_offset
            print("\n***************************************")
            # sphero.set_rgb(0, 0, 0)
            log = []
            log_i = step_det(sphero, cam, angle_offset, display, screen)
            # print("Log output: ", log_i) 
            print("        Pixel coords: ", log_i[0])
            print("        Angle offset: ", (angle_offset)) 
            print("Grid angle to target: ", log_i[1]) 
            print("Sent heading command: ", log_i[2])
            print("***************************************************")
            log.append(log_i)
            for i in range(n-1):
                time.sleep(2)
                log_i=step_det(sphero, cam, angle_offset, display, screen)
                print("        Pixel coords: ", log_i[0])
                print("        Angle offset: ", (angle_offset))
                print("Grid angle to target: ", log_i[1] * 180/np.pi) 
                print("Sent heading command: ", log_i[2])
                print("***************************************************")
                time.sleep(2)
                log.append(log_i)
            sphero.set_rgb(0, 0, 0xFF)
            print("Episode complete\n")
            sphero.roll(randint(20, 50), randint(0,359))
            time.sleep(2)
        print("Game over. Everyone wins!")

def dry_run(cam, n = 10, episodes = 10):
    pygame.display.init()
    screen = pygame.display.set_mode((800, 600))
    white = (255, 64, 64)
    _ = input("Press Ctrl-D to abort, or any other key to continue training")
    for _ in range(episodes):
        _ = step_det(cam = cam, sphero = None, screen = screen)
        for i in range(n-1):
            _ = step_det(cam  = cam, sphero = None, screen = screen)


def one_image(i = 1):
    cam = cam_setup(i)
    _ = capture_image(cam)
    labelled_image, image, _ = segment_photo_bmp()
    filtered_regions = filter_regions(labelled_image)
    print(observe_state(filtered_regions))
    cam_quit(cam)


if __name__ == "__main__":
    one_image()
    addrs = ['68:86:E7:06:FD:1D', '68:86:E7:07:07:6B', '68:86:E7:08:0E:DF']
    # cam = cam_setup(1)
