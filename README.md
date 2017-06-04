# Sphero.Swarms

An exploration of swarm robotics. The base development swarm is three Sphero SPRK's, and a webcam. For initial tasks, I use a "hive mind" model, with a single neural net controlling the whole swarm. In the future, I'd like to have a development swarm with the capacity for ad-hoc cooperation. That is, a neural network for each agent in the swarm.

# Initial Goals

The goals are behavioral in nature, rather than statistic.
- ** Simulation ** : demonstrate reinforcement learning for swarms
- ** Navigation ** : using reinforcement learning
- ** Coordination ** : collective navigation
- ** Cooperation ** : swarm moves a target one could not move on its own

# Major utilities:

- [Python.Swarms](https://github.com/elmar-hinz/Python.Swarms): Swarm simulation  
- [kulka](https://github.com/karol-szuster/kulka): Sphero Python API
- [SKImage](http://scikit-image.org/): Scikit Learn compatible image processing
- [openCV]
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
