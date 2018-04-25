'''
A training engine specifically for Sphero control q-table.
'''
import time
import numpy as np

class Q_Learner():
    def __init__(self, actions = 5):
        # Q learner
        self.q_table = dict()
        self.actions = actions
        self.reset_memory()

    def reset_memory(self):
        self.memory = {"inputs": [], "choices": [], "rewards": []}
        self.epsilon = 0.1

    def get_reward(self, reward):
        self.memory['rewards'].append(reward)

    def step(self, s_prime, choice = None):
        # make predictions, the inputs are required to step from
        # we say we are always in step s_prime, after executing action a

        # lookup the quality values
        if str(s_prime) in self.q_table.keys(): # check if we've seen this state
            quality = self.q_table[str(s_prime)] # grab the values
        else: # if not, generate some random numbers with mean zero
            quality = 2*np.random.random((1,self.actions)) - 1

        # perform training step
        if len(self.memory['inputs']) > 0:
            s = self.memory['inputs'][-1]
            a = self.memory['choices'][-1] # these are -1 because the game gives rewards
            r = self.memory['rewards'][-1] # after the figure.step() has occured
            if str(s) in self.q_table.keys(): # check if we've seen this state
                target = self.q_table[str(s)] # grab the values
            else: # if not, generate some random numbers with mean zero
                target = 2*np.random.random((1,self.actions)) - 1
            gamma = 0.1
            target[0][a] = r + gamma * np.max(quality[0])
            self.q_table[str(s)] = target
        # store this frames' input
        self.memory['inputs'].append(s_prime)

        if choice == None:
            d = np.random.random()
            # explore some of the time
            if d < self.epsilon:
                choice = np.random.randint(0, 4)
            # exploit current Q-function
            else:
                choice = np.argmax(quality)

        self.memory['choices'].append(choice) # store a'
        return choice # return a' to world

class DistGame():
    # linear "navigation" game
    # agent starts at 0, must get to 10
    # agent chooses speed between 0 and 4
    # reward is -0.01 for each step, 1 for each step at 10,
    # game is 10 steps long, ends with large penalty if agent overshoots
    def __init__(self, max_speed = 5):
        # make the Q-learner
        self.max_speed = max_speed
        self.player = Q_Learner(max_speed)
        self.player.position = 0

    def step(self, verbose = False):
        # get state
        state = self.player.position

        # choose action
        action = self.player.step(state)

        # update state
        movement = -1*self.max_speed
        self.player.position += movement
        s_prime = self.player.position

        # give rewards
        reward = 0
        if s_prime == 0:
            reward += 1
        elif s_prime < 0:
            reward += -100
        else:
            reward += -0.1
        self.player.get_reward(reward)

        # status update
        if verbose == True:
            # import pdb; pdb.set_trace()
            print("Step %d, s: %d, a: %d, r: %.2f, s': %d" % (i, state, action, reward, s_prime))

    def play(self, verbose = False):
        self.player.position = np.random.randint(10*self.max_speed)
        for i in range(10):
            self.step(verbose)
        self.player.reset_memory()
        self.player.position = 0

class AngleGame():
    # angular "navigation" game
    # agent starts at randint(0, 359), must get to 0
    # agent chooses speed between 0 and max_speed
    # reward is -0.01 for each step, 1 for each step at goal,
    # game is 10 steps long
    def __init__(self, max_speed = 5, directions = 36):
        # make the Q-learner
        self.max_speed = max_speed
        self.player = Q_Learner(2*max_speed+1)
        self.player.position = 0

    def step(self, verbose = False):
        # get state
        state = self.player.position

        # choose action
        action = self.player.step(state)

        # update state
        movement = action - self.max_speed
        self.player.position += movement
        self.player.position = self.player.position % 36
        s_prime = self.player.position

        # give rewards
        reward = 0
        if s_prime == 0:
            reward += 1
        else:
            reward += -0.1
        self.player.get_reward(reward)

        # status update
        if verbose == True:
            # import pdb; pdb.set_trace()
            print("s: %d, a: %d, r: %.2f, s': %d" % (state, action, reward, s_prime))

    def play(self, verbose = False):
        self.player.position = np.random.randint(35)
        for i in range(10):
            self.step(verbose)
        self.player.reset_memory()
        self.player.position = 0

class MultiGame():
    # "navigation" game
    # agent starts at randint(0, 10), randint(0, 359), must get to (0, 0)
    # agent chooses speed between 0 and max_speed
    # reward is -0.01 for each step, 1 for each step at goal,
    # game is 10 steps long
    def __init__(self, max_speed = 25, directions = 36):
        # make the Q-learner(s)
        self.max_speed = max_speed
        self.angle = AngleGame(max_speed, directions)
        self.distance = DistGame(max_speed)

    def step(self, verbose = False):
        state = (self.angle.player.position, self.distance.player.position)
        self.angle.step()
        self.distance.step()
        s_prime = (self.angle.player.position, self.distance.player.position)
        # status update
        if verbose == True:
            # import pdb; pdb.set_trace()
            print("s: %s, s': %s" % (str(state), str(s_prime)))

    def play(self, verbose = False):
        self.angle.player.position = np.random.randint(360)
        self.distance.player.position = np.random.randint(10*self.max_speed)
        for i in range(10):
            self.step(verbose)
        self.angle.player.reset_memory()
        self.distance.player.reset_memory()
        self.angle.player.position = 0
        self.distance.player.position = 0

if __name__=='__main__':
    global_start_time = time.time()
    game = MultiGame()
    episodes = 100000
    for i in range(episodes):
        if (i+1)%(episodes/100)==1:
            game.angle.player.epsilon = 0
            game.distance.player.epsilon = 0
            print("Game ", i+1)
            game.play(True)
            epsilon = (2*episodes - i)/2*episodes
            game.angle.epsilon = epsilon
            game.distance.epsilon = epsilon
        else:
            game.play()
