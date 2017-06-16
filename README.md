[Supervisedlearningcurve]: https://github.com/thetabor/Neural.Swarms/blob/master/notes/images/supervised_curve_0.png
{:height="300px" width="300px"}

# Neural.Sphero
![Segmentation test](https://raw.githubusercontent.com/thetabor/Sphero.Swarms/master/image_processing/images/segmentation/sphero_seg2.png)

An exploration of reinforcement learning with Deep-Q Networks using Sphero robotic balls.

# Initial Goals

The goals are behavioral in nature, rather than statistic.
- **Simulation** : demonstrate reinforcement learning for simple navigation
- **Navigation** : using reinforcement learning in a physical space
- **Cooperation** : multiple agents move a target

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

# Agents, Environments, and Reinforcement Learning

- **Agents**, in the context of machine learning, are a class of algorithms which make choices once deployed. These include everything from humble vacuum cleaners to stock-picking algorithms.
- **Supervised Learning** is the standard method of many statistical models and neural networks. It requires an (X, y) style training set, with inputs and desired outputs. For any kind of agent, this becomes a limitation, as the agent will be limited by the input data. That being said, we demonstrate supervised learning in the [Neural.Swarms](https://github.com/thetabor/Neural.Swarms) simulation. For the simple task of reaching a goal position in a deterministic environment, it performs very well after a short training period. We obtain data for this training from a deterministic strategy, so the neural network is limited to that performance level on the game. Simply adding a small barrier is enough to reduce the agents' performance.
- **Reinforcement Learning** is a method of training machine learning algorithms with rewards. Rather than learning from an (X, y) training set, it learns from experience. Each experience comes with certain rewards, and each time a reward is received, the algorithm can learn.
- **Deep-Q Networks** are a way to deploy reinforcement learning to neural networks. The network predicts Q-values for each action the network is allowed. A Q-value is a **quality** of a state, or the expected sum of rewards as we play the game from that state. We (almost) always select the max Q-value we predict.
- **RL Data:** Initially, the agent has absolutely no knowledge of the environment, so Q-values are effectively random. At each step, it updates the Q-value using the actual reward, plus the Q-value of the next step it plans on taking. So, our model fits (X, y) data, but each y is actually self-generated and often very inaccurate. But since a part of it is ground truth, the model eventually learns something close enough to real Q-values to function.

# Simulation

Using the [Neural.Swarms](https://github.com/thetabor/Neural.Swarms) simulation engine I can implement either supervised or reinforcement learning. Here we see how quickly the supervised learner can perform well on the simple task. In contrast, the reinforcement learner struggles to perform well, but it is does show potential. Here are some examples of simulation performance:

| Deterministic Strategy | Almost trained supervised model | Trained supervised |
| --- | --- | --- |
| ![Deterministic](https://github.com/thetabor/Neural.Swarms/blob/master/notes/gifs/deterministic_strategy_test.gif) | ![Almost trained](https://github.com/thetabor/Neural.Swarms/blob/master/notes/gifs/supervised/slight_undertrained_supervised.gif) | ![Fully trained](https://github.com/thetabor/Neural.Swarms/blob/master/notes/gifs/supervised/trained_supervised.gif) |

The supervised network learns from the deterministic strategy on the left, and eventually learns to mimic it perfectly.

| Typical supervised learning curve |
| --- |
|![Supervisedlearningcurve]|

This also means that the supervised learner is limited by the strategy it learns from. So, enter *reinforcement learning*!

Reinforcement learning allows the agent to explore strategies on its own, and by receiving rewards from its environment, learns which are better. With reinforcement learning, I've struggled to get good results on the large grid, so I focused on a small game for now.

| RL early training | RL mid training | RL late training |
| --- | --- | --- |
| ![RL1](https://github.com/thetabor/Neural.Swarms/blob/master/notes/gifs/three_stages_rl/trained_guided_rl_1.gif) | ![RL2](https://github.com/thetabor/Neural.Swarms/blob/master/notes/gifs/three_stages_rl/trained_guided_rl_2.gif) | ![RL3](https://github.com/thetabor/Neural.Swarms/blob/master/notes/gifs/three_stages_rl/trained_guided_rl_3.gif) |

Above we see the progression of the agents learning. In the first, it had seen about 50,000 game steps. The next was an additional 500,000, and the final saw  another 1,000,000 steps. By contrast, the supervised learner above only required about 10,000 steps to achieve nearly-perfect imitation of the deterministic strategy. So why bother with reinforcement learning? I discuss below. First, here are some very noisy charts from the middle and right hand training cycles.

| 500,000 steps | Another 1,000,000! |
| --- | --- |
| ![RL1](https://github.com/thetabor/Neural.Swarms/blob/master/notes/images/rl_plots9x9_500000_2_4_Adam.png) | ![RL2](https://github.com/thetabor/Neural.Swarms/blob/master/notes/images/rl_plots9x9_1000000_3_4_Adam.png) |

Here's a good example of RL not working:

| ~ 3 millions steps |
| :---: |
| ![RL1](https://github.com/thetabor/Neural.Swarms/blob/master/notes/gifs/bad_rl/RL_after_12_rounds.gif) |
| Final 1,000,000 steps of training |
| ![sadchart](https://github.com/thetabor/Neural.Sphero/blob/master/notes/rl_plots9x9_1000000_3_4_Adam.png) |

# Future sections
- Base network architecture
- Training program
- Computer vision
- Decision making
