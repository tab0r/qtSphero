# Sphero.Swarms

An exploration of swarm robotics. The base development swarm is three Sphero SPRK's, and a webcam. For initial tasks, I use a "hive mind" model, with a single neural net controlling the whole swarm. In the future, I'd like to have a development swarm with the capacity for ad-hoc cooperation. That is, a neural network for each agent in the swarm.

# Initial Goals

The goals are behavioral in nature, rather than statistic.
- **Simulation** : demonstrate reinforcement learning
- **Navigation** : using reinforcement learning in a physical space
- **Coordination** : collective navigation
- **Cooperation** : swarm moves a target one could not move on its own

# Major utilities:

- [Python.Swarms](https://github.com/elmar-hinz/Python.Swarms): Swarm simulation  
- [kulka](https://github.com/karol-szuster/kulka): Sphero Python API
- [SKImage](http://scikit-image.org/): Scikit Learn compatible image processing
- [pygame](https://www.pygame.org/news): pygame.camera for webcam image capture
- [Keras](https://keras.io/): Neural network frontend
- [theano](https://github.com/Theano/Theano): Neural network backend

Should run on most Unix (Linux, Mac) but probably not directly on Windows.
Coded with Python 3.6 on Mac OS X and Lubuntu 16ish.

# References

- Excellent materials in Georgia Tech's [Reinforcement Learning](https://www.udacity.com/course/reinforcement-learning--ud600) course on Udacity.
- Referencing Karpathy's blog in two places:
    - [Keras Plays Catch](https://edersantana.github.io/articles/keras_rl/)
    - [Pong from Pixels](http://karpathy.github.io/2016/05/31/rl/)
- Nervanasys blog post (linked in Karpathy):
    - [Deep Reinforcement Learning](https://www.nervanasys.com/demystifying-deep-reinforcement-learning/)

# Methods & Data

- **Supervised Learning** is the standard method of many statistical models and neural networks. It requires an (X, y) style training set, with inputs and desired outputs. For any kind of agent, this becomes a limitation, as the agent will be limited by the input data. That being said, we demonstrate supervised learning in the [Neural.Swarms](https://github.com/thetabor/Neural.Swarms) simulation. For the simple task of reaching a goal position in a deterministic environment, it performs very well after a short training period. We obtain data for this training from a deterministic strategy.
- ** Reinforcement Learning ** is a method of training machine learning algorithms with rewards. Rather than learning from an (X, y) training set, it learns from experience. Each experience comes with certain rewards, and each time a reward is received, the algorithm can learn. A Deep-Q Network implements this method as neural network Q-estimator. A thorough explanation of the DQN and Q-estimation is available in the [references](https://www.nervanasys.com/demystifying-deep-reinforcement-learning/) above.
- **RL Data:** At each step of the game, the agent experiences something and learns from it. Exactly what it is learning requires some subtle ideas. The reward is a 'ground truth' on which the agent can based its understanding of the world. But, we're interested in more than short-term rewards. We'd like to encourage the agent into complex, intelligent behavior. So, it must look forward. Enter the Q-value. A Q-value is a ** quality ** of a state, or the expected sum of rewards as we play the game from that state. Initially, the agent has absolutely no knowledge of the environment, so Q-values are effectively random. At each step, it updates the Q-value using the actual reward, plus the Q-value of the next step it plans on taking. So, our model fits (X, y) data, but each y is actually self-generated and often very innacurate. But since a part of it is ground truth, the model eventually learns something close enough to real Q-values to funtion.
-![Segmentation test](https://raw.githubusercontent.com/thetabor/Sphero.Swarms/master/image_processing/images/segmentation/sphero_seg2.png)
-![Game test](https://raw.githubusercontent.com/thetabor/Sphero.Swarms/master/image_processing/notes/game_test.png)
- **Simulation** Using the [Neural.Swarms](https://github.com/thetabor/Neural.Swarms) simulation engine I can implement either supervised or reinforcement learning. Here we see how quickly the supervised learner can perform well on the simple task. In contrast, the reinforcement learner does not yet perform well.
