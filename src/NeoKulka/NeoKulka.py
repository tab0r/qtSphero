'''
A training engine specifically for Sphero control DQN.
Make data generates data that looks like actual data.
Hard coded to my base variables:
grid size: (640, 480)
    -inputs will always be within this grid
    -actual position allowed a radius r from initial point
    -actual position can be outside the grid
data inputs: 5 coordinate pairs plus the command that generated them
cmd outputs: 9 outputs,
    -8 for stay
    -0-3 for low speed at 90 degree angles
    -4-7 for high speed at 90 degree angles
'''
import time, os, sys

def make_episode(p = 1, steps = 5):
    '''
    generate new training episode
    store as pickle
    p is a control parameter for stochasticity
    '''
    offset = randint(0, 359)
    pass

def data_gen(p = 1, episodes = 1000, steps = 5):
    for _ in range(episodes):
        make_episode(p, steps)

def train_model(model, folder_path, limit = 10, limit_mode = "time"):
    '''
    train the model on data in folder_path
    assumes files are episodes of training
    randomly selects until limit reached
    by default, limits training time to ten minutes
    '''
    if limit_mode == "time":
        t0 = time.time()
    else:
        print("Not yet implemented")
