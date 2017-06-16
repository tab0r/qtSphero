# in this, we attempt to train a Sphero to stay in the center of our view.
from deterministic_sphero import *

# neural network imports
from keras.utils import np_utils
from keras.models import Sequential, load_model
from keras.layers.core import Dense
from keras.optimizers import sgd, RMSprop, Adagrad, Adam

def step_game(sphero, cam, e = 0.1, model = None, last_input = None, last_choice = None):
    _ = capture_image(cam)
    labelled_image, image, _ = segment_photo_bmp()
    filtered_regions = filter_regions(labelled_image)
    data = observe_state(filtered_regions)
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
            target = np.zeros(5)
            target[last_choice] = reward + predicts[0][last_choice]
            loss += model.train_on_batch(last_input, np.array(target).reshape(1,5))
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
    print("Q-choice: ", np.argmax(predicts))
    print("Reward: ", reward)
    print("Distance: ", distance)
    if sphero != None:
        choice = np.argmax(predicts)
        if np.random.rand() < e:
            choice = randint(0,4)
            if choice != 4:
                direction = choice * 90
                speed = 28
                sphero.roll(speed, direction)
    return inputs, predicts, reward, distance

def baseline_model(optimizer = Adam(),
                    layers = [{"size":600,"activation":"relu"}]):
    inputs = 1200
    num_outputs = 360
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
    model.add(Dense(num_outputs, activation='tanh'))
    model.compile(optimizer = optimizer,
                    loss = "mean_squared_error")
    return model

# '68:86:E7:06:FD:1D', '68:86:E7:07:07:6B', '68:86:E7:08:0E:DF'
def play(cam, addr, n = 10, episodes = 10, mode = 0, display = True):
    if os.path.isfile('models/sphero_0.h5') == True:
        model = load_model( "models/sphero_0.h5")
        print("Saved model loaded")
    else:
        model = baseline_model()
    if display == True:
        pygame.display.init()
        screen = pygame.display.set_mode((800, 600))
        white = (255, 64, 64)
    print("Bringing Sphero online")
    with Kulka(addr) as sphero:
        print("Sphero online!")
        # spheros.append(sphero)
        sphero.set_inactivity_timeout(300)
        sphero.set_rgb(0, 0, 0x0F)
        for epi in range(episodes):
            print("Beginning episode ", epi)
            # for sphero in spheros:
            sphero.set_rgb(0xFF, 0xFF, 0xFF)
            # pdb.set_trace()
            print("Current offset angle: ", angle_offset)
            log = []
            log_i = step_game(sphero, cam, angle_offset, display, screen)
            # print("Log output: ", log_i) 
            print("        Pixel coords: ", log_i[0])
            print("        Angle offset: ", (angle_offset)) 
            print("Grid angle to target: ", log_i[1]) 
            print("Sent heading command: ", log_i[2])
            print("***************************************************")
            log.append(log_i)
            for i in range(n-1):
                time.sleep(2)
                log_i=step_game(sphero, cam, angle_offset, display, screen)
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
    sphero.close()
    # cam_quit(cam)
    model.save("models/sphero_0_p.h5" )
    print("Prime model saved")

if __name__ == "__main__":
    one_image()
    addrs = ['68:86:E7:06:FD:1D', '68:86:E7:07:07:6B', '68:86:E7:08:0E:DF']
    cam = cam_setup(1)
