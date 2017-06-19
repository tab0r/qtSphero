# Neural.Orb
![Segmentation test](https://raw.githubusercontent.com/thetabor/Sphero.Swarms/master/image_processing/images/segmentation/sphero_seg2.png)

An exploration of reinforcement learning with Deep-Q Networks using Sphero robotic balls.

# Sections

This readme is a bit long, but describes my development process to my current states.

- Initial Goals
- Agents, Environments, and Reinforcement Learning
- Simulation
- Deploying to Sphero
- Computer vision
- Rewards and sparsity
- Why reinforcement learning?
- Base network architecture
- Making a reinforcement learner learn faster(er)
- Simulations, reloaded
- Technical difficulties
- Major utilities
- References

# Initial Goals

The goals are behavioral in nature, rather than statistic.
- **Simulation** : demonstrate reinforcement learning for simple navigation
- **Navigation** : using reinforcement learning in a physical space
- **Cooperation** : multiple agents move a target

# Agents, Environments, and Reinforcement Learning

- **Agents**, in the context of machine learning, are a class of algorithms which make choices once deployed. These include everything from humble vacuum cleaners to stock-picking algorithms.
- **Supervised Learning** is the standard method of many statistical models and neural networks. It requires an (X, y) style training set, with inputs and desired outputs. For any kind of agent, this becomes a limitation, as the agent will be limited by the input data. That being said, we demonstrate supervised learning in the [Neural.Swarms](https://github.com/thetabor/Neural.Swarms) simulation. For the simple task of reaching a goal position in a deterministic environment, it performs very well after a short training period. We obtain data for this training from a deterministic strategy, so the neural network is limited to that performance level on the game. Simply adding a small barrier is enough to reduce the agents' performance.
- **Reinforcement Learning** is a method of training machine learning algorithms with rewards. Rather than learning from an (X, y) training set, it learns from experience. Each experience comes with certain rewards, and each time a reward is received, the algorithm can learn.
- **Deep-Q Networks** are a way to deploy reinforcement learning to neural networks. The network predicts Q-values for each action the network is allowed. A Q-value is a **quality** of a state, or the expected sum of rewards as we play the game from that state. We (almost) always select the max Q-value we predict.
- **RL Data:** Initially, the agent has absolutely no knowledge of the environment, so Q-values are effectively random. At each step, it updates the Q-value using the actual reward, plus the Q-value of the next step it plans on taking. So, our model fits (X, y) data, but each y is actually self-generated and often very inaccurate. But since a part of it is ground truth, the model eventually learns something close enough to real Q-values to function.

# Simulation

Using the [Neural.Swarms](https://github.com/thetabor/Neural.Swarms) simulation engine I can implement either supervised or reinforcement learning. Here we see how quickly the supervised learner can perform well on the simple task. In contrast, the reinforcement learner struggles to perform well, but it is does show potential. Here are some examples of simulation performance:

| Deterministic Strategy | Almost trained supervised model | Trained supervised | Supervised curve |
| --- | --- | --- | --- |
| ![Deterministic](https://github.com/thetabor/Neural.Swarms/blob/master/notes/gifs/deterministic_strategy_test.gif) | ![Almost trained](https://github.com/thetabor/Neural.Swarms/blob/master/notes/gifs/supervised/slight_undertrained_supervised.gif) | ![Fully trained](https://github.com/thetabor/Neural.Swarms/blob/master/notes/gifs/supervised/trained_supervised.gif) | ![Supervised learning curve](https://github.com/thetabor/Neural.Swarms/blob/master/NaviGame/supervised_curve_1.png) |

The supervised network learns from the deterministic strategy on the left, and eventually learns to mimic it perfectly.

This also means that the supervised learner is limited by the strategy it learns from. So, enter **reinforcement learning**! Reinforcement learning allows the agent to explore strategies on its own, and by receiving rewards from its environment, learns which are better. With reinforcement learning, I've struggled to get good results on the large grid, so I focused on a small game at first.

| RL early training | RL mid training | RL late training |
| --- | --- | --- |
| ![RL1](https://github.com/thetabor/Neural.Swarms/blob/master/notes/gifs/three_stages_rl/trained_guided_rl_1.gif) | ![RL2](https://github.com/thetabor/Neural.Swarms/blob/master/notes/gifs/three_stages_rl/trained_guided_rl_2.gif) | ![RL3](https://github.com/thetabor/Neural.Swarms/blob/master/notes/gifs/three_stages_rl/trained_guided_rl_3.gif) |

Above we see the progression of the agents learning. In the first, it had seen about 50,000 game steps. The next was an additional 500,000, and the final saw  another 1,000,000 steps. By contrast, the supervised learner above only required about 10,000 steps to achieve nearly-perfect imitation of the deterministic strategy. So why bother with reinforcement learning? I'll revisit that issue. First, here are some very noisy charts from the middle and right hand training cycles, and then some actual discussion of the Sphero.

| 500,000 steps | Another 1,000,000! |
| --- | --- |
| ![RL1](https://github.com/thetabor/Neural.Swarms/blob/master/notes/images/rl_plots9x9_500000_2_4_Adam.png) | ![RL2](https://github.com/thetabor/Neural.Swarms/blob/master/notes/images/rl_plots9x9_1000000_3_4_Adam.png) |

# Deploying to Sphero

With a good supervised model, and some progress on RL, I wanted to get my agents controlling the robot as soon as possible. To say it's easier said than done... Is an understatement. The first challenge was getting a connection to Sphero in Python; something Sphero's tech support will insist is not possible. Thankfully, people love doing things that are not possible, and the [kulka](https://github.com/karol-szuster/kulka) library provided everything I needed to send commands to Sphero. I added a data polling function to it (see my fork of [kulka](https://github.com/thetabor/kulka)). But, this didn't get me very far. Without full sensor streaming, the data was too limited to get where I wanted, and in the timeline of this project, I needed good sensor data without having to learn a new technology, so I switched to computer vision.

# Computer vision

This is not a field which particularly interests me, there are plenty of brilliant, talented, minds working on it, and the theory is too convoluted (heh) to really interest me. Thankfully, those brilliant, talented minds provide cool things like [SKImage](http://scikit-image.org/). In ```src/webcam_segmentation/py``` you'll find the methods I used, all very standard packages in the SKImage library. With a filter and blob detection, I found the Sphero in the images reliably enough. The filtered image became my neural network inputs for some tests, and in other tests I just used the coordinate as inputs.

| Equalizations | Histogram Equalizations | Contrast Adjustment |
| --- | --- | --- |
| ![eq](https://github.com/thetabor/Neural.Sphero/blob/master/image_processing/images/filtering/eq_test.png) | ![hist_eq](https://github.com/thetabor/Neural.Sphero/blob/master/image_processing/images/filtering/hist_eq_test.png) | ![contrast](https://github.com/thetabor/Neural.Sphero/blob/master/image_processing/images/filtering/contrast_adjustment.png) |

Ultimately, gamma correction (middle of right image) on a massively downsampled image provided a consistent enough reading for both neural network inputs and for scoring the agent.

| ex. 1 | ex. 2 | ex. 3 |
| --- | --- | --- |
| ![img1](https://github.com/thetabor/Neural.Orb/blob/master/notes/last_frame.png) | ![img2](https://github.com/thetabor/Neural.Orb/blob/master/notes/last_frame_archive_1.png) | ![img3](https://github.com/thetabor/Neural.Orb/blob/master/notes/last_frame_archive_0.png) |

This brings us too...

# Rewards and sparsity

For the simulation models the agent receives rewards on [-1, 1] based on distance from the goal. This helped the agents learn faster than in other contexts, as each state had a distinct reward associated with it. In other domains, like the DeepMind network which plays Atari games, many actions have no rewards associated with them until the agent reaches an end state. This is called the *credit assignment problem*. If the agent receives rewards only at the end of the game, how can it tell which actions led to that reward? Reinforcement learners solve this through massive amounts of gameplay training, and the Q-function simply learns to implicitly calculate sequences of rewards. In simulation, I can avoid the credit assignment problem entirely with the reward function I've chosen. But, when I deploy that to Sphero, the CV algorithms aren't reliable enough to always give a reward, and sometimes give totally incorrect rewards. Without training an entirely new model just for scoring the RL model, there's no way around this, and the Sphero agent will have to work its way through the credit assignment problem through brute force.

# Why reinforcement learning?

While I struggled to get my RL agents to perform as well their supervised counterparts, their versatility is evident. As we can see above, the supervised learner achieves the results of the deterministic strategy, and no more. Notice how it even follows the 45 degree lines used by the deterministic strategy. Theoretically, our RL agent could find the shorter "straight line" path. Additionally, if you can achieve your results with a deterministic strategy, why bother with a neural network at all?

More specifically, in the context of running a Sphero, a deterministic strategy is possible, assuming something very important. When a Sphero turns on, it makes its current heading the "zero" heading. Assuming we know that direction relative to our image's notion of a "zero" heading, then getting the Sphero where we want it is deterministically possible. In fact, in ```src/deterministic_sphero.py``` you'll find my best efforts at doing exactly that.

I did not get far.

And that, is why reinforcement learning.

# Base network architecture

All work has been done with standard multi-layer perceptrons. The beauty of reinforcement learning is that it doesn't require anything complicated in neural structure, simply the right reward system.

I am exploring many kinds of inputs to my neural nets:

- Coordinates of agent and goal (4 inputs)
- Preprocessed 40x30 image (1200 inputs)
- Image with coordinates (1202 - 1204 inputs)
- not yet implemented: N stacked coordinates (Nx2 inputs)

In each case, I focus on training networks with no more than five layers, including input and output. The supervised learner performs very well with just two hidden layers with 20 neurons each, on any of the first three input types. So, I have no reason to build bigger networks until we see to limit of these networks under reinforcement learning.

# Making a reinforcement learner learn faster(er)

Given the issues above, I need to speed up the Sphero control model as much as possible. By taking advantage of deterministic strategies as much as possible, I attempt to get training time low enough to deploy physically. But, when a physical training step takes 5-10 seconds, performing enough of those steps to learn will be very, very time-intensive. So much so, the use of neural networks in this context becomes questionable.

Let's do it anyway.

To be clear, the goal here to **get a Sphero to consistently navigate to the center of our camera view**. Accomplishing this opens up all kinds of possibilities for navigation and movement extensions that all other goals are sort of irrelevant. Accomplishing this goal involves solving the problem discussed in the [Deploying to Sphero](#Deploying-to-Shero) section. Ideally, we would be able to point the camera in different places, at many different angles, which makes the problem that much harder. If we had perfect alignment between our image and the Sphero zero angles, a deterministic strategy will work. So, we'll build a DQN which finds the offset angle needed for correct navigation.

Hopefully.

# Simulations, reloaded

So far, I've established that I can get an agent to learn through reinforcements, and set up a training environment for Sphero that takes far too long. To really dig into speeding up learning, let's return to the simulated environment.

When the DQN agent is initialized, it's output values are effectively random numbers, and training is very susceptible to local minima. So, we train using an explore/exploit ratio that decreases throughout the training session. Typically, it starts at 0.9, and ends at 0.1. Additionally, I can make some of the choices come from our deterministic strategy, to focus training on the "correct" routes. Third, we know that our deterministic strategy works, so why not use it? And, finally, a tolerance function can make the game easier or harder, to let's start with an easier game, then make it harder once the agent is doing well.

With all this in mind, I built a new model. This model takes inputs as usual, the whole game screen. As outputs, it has the five usual outputs; up, down, left, right and stay, plus a new addition: use the deterministic strategy. So, for the simple games, all our DQN agent has to do is learn to always use the deterministic strategy. Once it learns this, then we can start exploring more complex problems. Meet Larry, the simple bundle of neurons:

| Break In | More Training | Trained with harder game | Non-optimal paths |
| --- | --- | --- | --- |
| ![Larry1](https://github.com/thetabor/Neural.Swarms/blob/master/NaviGame/larry/20000_x_15/larry_gif_00_20000x15.gif) | ![Larry2](https://github.com/thetabor/Neural.Swarms/blob/master/NaviGame/larry/20000_x_15/larry_gif_01_plus60000x5_a.gif) | ![img3](https://github.com/thetabor/Neural.Swarms/blob/master/NaviGame/larry/20000_x_15/larry_gif_07_plus_2x_60000x5_d.gif) | ![img4](https://github.com/thetabor/Neural.Swarms/blob/master/NaviGame/larry/20000_x_15/larry_gif_09_plus_3x_60000x5_b.gif) |

Larry was doing very well. These plots show how certain patterns recur in the course of training:

| Break In | More Training | Training with harder game |
| --- | --- | --- |
| ![Larry1](https://github.com/thetabor/Neural.Swarms/blob/master/NaviGame/larry/20000_x_15/larry_plot_0_t5.png) | ![Larry2](https://github.com/thetabor/Neural.Swarms/blob/master/NaviGame/larry/20000_x_15/larry_plot_1_t5.png) | ![img3](https://github.com/thetabor/Neural.Swarms/blob/master/NaviGame/larry/20000_x_15/larry_plot_2_t2.png) |

All this progress made me wonder if Larry could handle a challenge...

| And he ran away screaming... | Training hasn't helped yet | Still works on simple games |
| --- | --- | --- |
| ![Larry1](https://github.com/thetabor/Neural.Swarms/blob/master/NaviGame/larry/20000_x_15/larry_gif_11_plus_3x_60000x5_d.gif) | ![Larry2](https://github.com/thetabor/Neural.Swarms/blob/master/NaviGame/larry/20000_x_15/larry_gif_12_plus_3x_60000x5_e.gif) | ![Larry3](https://github.com/thetabor/Neural.Swarms/blob/master/NaviGame/larry/20000_x_15/larry_gif_14_larry_maze_d.gif)

# Technical difficulties

Unfortunately, this is not a success story. It is an ongoing one, which I hope to make a success story.

Training a physical robot takes a lot of time, and depends on a lot of things. Progress is pretty much halted here. I cannot run training consistently and long enough to get positive results, but the backend is there. I haven't implemented use of N stacked coordinates, but I think that will give me something useful. This project taught me a lot, and I hope to continue development because I love the Sphero robot, and I think the applications to other robots will be awesome (quadcopter formations, anyone?). If you've read all this, thank you for you time. Here's a brief list of the reasons why I'm stopping here:

- computer instability (training big neural nets can burn up your CPU)
- no option to upload to a more powerful/stable machine
- Bluetooth connection instability
- lack of experience with streaming Bluetooth data
- CV failures, creating reward sparsity and slowing training
- CV errors, creating incorrect rewards and sometimes reversing training
- utility incompatibility (Bluetooth not available for Python on macOS)

# Next steps and future sections

I'll continue to develop this as much as possible. I love the Sphero robot and the possibilities it provides to the educational and open source communities. I'll continue to find ways to train the offset DQN as I feel it has a lot of potential. It also demonstrates a powerful method of model extension. In this case, I'm trying to extend a deterministic model with a reinforcement learning neural network, but the method applies to many kinds of models. Instead of a deterministic base, we could've use the supervised model I showed above. We could also have multiple deterministic models for the DQN to select from. Ultimately, developing an agent which can adapt means we need a system of model extension, and this project demonstrates a powerful method of doing that.

If the model can start training consistently, I can record the training runs, and start training the model offline. But, that will require a break in, as early data will be massively skewed by the randomly initialized DQN.

Thank you again for reading this far! Have fun. :)

# Major utilities:

- [Python.Swarms](https://github.com/elmar-hinz/Python.Swarms): Swarm simulation  
- [kulka](https://github.com/karol-szuster/kulka): Sphero Python API
- [SKImage](http://scikit-image.org/): Scikit Learn compatible image processing
- [pygame](https://www.pygame.org/news): pygame.camera for webcam image capture
- [Keras](https://keras.io/): Neural network frontend
- [theano](https://github.com/Theano/Theano): Neural network backend
- [TensorFlow](https://www.tensorflow.org/): Neural network backend

Should run on most Unix (Linux, Mac) but probably not directly on Windows.
Coded with Python 3.6 on Mac OS X and Lubuntu 16ish.

# References

- Excellent materials in Georgia Tech's [Reinforcement Learning](https://www.udacity.com/course/reinforcement-learning--ud600) course on Udacity.
- Referencing Karpathy's blog in two places:
    - [Keras Plays Catch](https://edersantana.github.io/articles/keras_rl/)
    - [Pong from Pixels](http://karpathy.github.io/2016/05/31/rl/)
- Nervanasys blog post (linked in Karpathy):
    - [Deep Reinforcement Learning](https://www.nervanasys.com/demystifying-deep-reinforcement-learning/)
