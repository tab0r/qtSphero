'''
This file generates data for supervised training of a neural network for Sphero.
The inputs are the five variables kulka.data_poll() returns:
    xpos, ypos, xvel, yvel, sog
where
    sog ~= sqrt(xvel**2 + yvel**2)
The outputs are speed, heading, and time.
'''
