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
import time, os, sys, pickle
from os import listdir
from os.path import isfile, join
from random import randint

# neural network imports
from keras.utils import np_utils
from keras.models import Sequential, load_model
from keras.layers.core import Dense
from keras.optimizers import Adam

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
    losses = []
    if limit_mode == "time":
        t0 = time.time()
        t1 = time.time()
        t_str = "Training for " + str(limit) + " minutes."
        step_count = 0
        count = 0
        print(t_str)
        while((t1 - t0) < (limit * 60)):
            count += 1
            # print("t1-t0:", t1-t0)
            if folder_path == None:
                mypath = "episodes/"
            else:
                mypath = folder_path
            filenames = [join(mypath, f) for f in listdir(mypath) if isfile(join(mypath, f))]
            choice = filenames[randint(0, len(filenames) - 1)]
            if choice == "episodes/.DS_Store":
                pass
            else:
                print("Filename: ", choice)
                output = pickle.load(open(choice, "rb"))
                step_count += len(output[0])
                print("input sample: ", output[0][0])
                print("target sample: ", output[1][0])
                if model != None:
                    replay_loss = model.train_on_batch(output[0], output[1])
                    losses.append(replay_loss)
                    model_filestr = "offline_training_backup.h5"
                    model.save(model_filestr)
                    print("loss on replay training: ", replay_loss)
                # time.sleep(1)
                t1 = time.time()
        end_str = "Complete. Saw " + str(step_count) + " steps in " + str(count) \
                    + " episodes."
        print(end_str)
        print("Execution time: ", t1-t0)
    else:
        print("Not yet implemented")
    return losses

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
    model = baseline_model()
    model.load_weights("sphero_model.h5")
    replay_log = train_model(model = model, folder_path = None, limit = 10)
    model.save("sphero_model.h5")
    # pickle.dump(open("replay_log.p", "wb"))
