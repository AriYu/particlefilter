#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#from numpy.random import *
import math
import matplotlib.pyplot as plt
import numpy

STATE_DIMENTION = 3 # x, y, yaw
MAX_RANGE = 10 #[m]

# RFIFのタグの位置
RFID=numpy.array([[10.0, 0],
                  [10.0, 10.0],
                  [0.0, 15.0],
                  [-5.0, 20],])

def process_model(x, u, dt):
    ret = [0.0] * STATE_DIMENTION
    ret[0] = x[0] + u[0]*dt*math.cos(x[2])
    ret[1] = x[1] + u[0]*dt*math.sin(x[2])
    ret[2] = x[2] + u[1]*dt
    return ret

def control_model(time):
    u = [0.0] * STATE_DIMENTION
    T = 10 #[sec]
    V = 1.0 #[m/s]
    yawrate = 5.0 #[deg/s]
    u[0] = V*(1-math.exp(-time/T))
    u[1] = math.radians(yawrate)*(1-math.exp(-time/T))
    return u

def observe_model(x, y):
    # マーカまでの距離を見て遠いのは使わないようにするのがいいかも
    # x : particleの位置・姿勢
    # y : 観測によって得られたRFIDまでの距離
    w = 1.0
    print(y)
    for i , rfid in enumerate(RFID):
        d = numpy.linalg.norm(rfid - x[:2]) # x[:2] means x[0], x[1]
        print(i)
        dz = y[i] - d
        w *= gaussian(dz, 0, 1.0)
    return w

def gaussian(x, u, sigma):
    return 1.0/math.sqrt(2.0*math.pi*sigma**2.0)*math.exp(-(x-u)**2.0/(2.0*sigma**2.0))


class Particle:
    """A simple Particle class"""

    def __init__(self, dimention):
        self.state = [0.0] * dimention
        self.weight = 0.0

    def print_particle(self):
        print(self.state)
        print(self.weight)


class ParticleFilter:
    """A simple ParticleFilter class"""

    def __init__(self, dimention, num_of_particles):
        self.particles = [Particle(dimention)] * num_of_particles
        self.num_of_particles = num_of_particles
        self.dimention = dimention

    def print_info(self):
        print("Dimention is {}".format(self.dimention))
        print("Number of particles is {}".format(len(self.particles)))
        print("Process Noise mean : {}".format(self.process_mean))
        print("Process Noise cov  : {}".format(self.process_cov))
        print("Observe Noise mean : {}".format(self.observe_mean))
        print("Observe Noise cov  : {}".format(self.observe_cov))

    def set_process_noise_param(self, mean, cov):
        self.process_mean = mean
        self.process_cov = cov

    def set_observe_noise_param(self, mean, cov):
        self.observe_mean = mean
        self.observe_cov = cov

    def predict(self, dt, process_function, control):
        process_noise = numpy.random.multivariate_normal(self.process_mean,
                                                         self.process_cov,
                                                         self.num_of_particles)
        for i, particle in enumerate(self.particles):
            self.particles[i].state = process_function(particle.state, control, dt)
            self.particles[i].state += process_noise[i]

    def sampling(self, observation, observe_function):
        for i, particle in enumerate(self.particles):
            self.particles[i].weight = observe_function(x=particle.state, y=observation)
        
    # def run(self):
    #     for time in range(10):
    #         print("time : {}".format(time))
    #         self.sampling(time, process_model, control_model)
    #         x = [x.state[0] for x in self.particles]
    #         y = [x.state[1] for x in self.particles]


if __name__ == "__main__":
    particlefilter = ParticleFilter(dimention=STATE_DIMENTION, num_of_particles=100)
    # process_mean = [0, 0, 0]
    # process_cov = [[1, 0, 0], [0, 1, 0], [0, 0, 1]]
    process_mean = numpy.array([0, 0, 0])
    process_cov = numpy.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
    particlefilter.set_process_noise_param(mean=process_mean, cov=process_cov)
    # observe_mean = [0]
    # observe_cov = [[1]]
    observe_mean = numpy.array([0])
    observe_cov = numpy.array([1])
    particlefilter.set_observe_noise_param(mean=observe_mean, cov=observe_cov)
    particlefilter.print_info()
    #    particlefilter.run()
    # truth_state = [0.0] * STATE_DIMENTION
    # deadr_state = [0.0] * STATE_DIMENTION # Dead reconing state
    truth_state = numpy.zeros((3, 1))
    deadr_state = numpy.zeros((3, 1))
    control = numpy.zeros((1, 2))
    observation = numpy.zeros((len(RFID), 1))
    dt = 0.1
    time = 0.0
    for i in range(100):
        # calculation ground truth
        time = time + dt
        control = control_model(time)
        truth_state = process_model(truth_state, control, dt)
        deadr_state = process_model(deadr_state,
                                    (control
                                     +numpy.random.multivariate_normal(process_mean,
                                                                       process_cov)),dt)
        particlefilter.predict(dt, process_model, control)
        for i , rfid in enumerate(RFID):
            observation[i] = (numpy.linalg.norm(RFID[i] - truth_state[:2])
                              + numpy.random.normal(observe_mean[0], observe_cov[0]))
        particlefilter.sampling(observation, observe_model)
