#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#from numpy.random import *
import math
import matplotlib.pyplot as plt
import numpy
from progressbar import ProgressBar, Percentage, Bar

STATE_DIMENTION = 3 # x, y, yaw
MAX_RANGE = 10 #[m]

# RFIFのタグの位置
RFID=numpy.array([[10.0, 0],
                  [10.0, 10.0],
                  [0.0, 15.0],
                  [-5.0, 20],])

def process_model(x, u, delta_time):
    ret = numpy.zeros((STATE_DIMENTION, 1))
    ret[0] = x[0] + u[0]*delta_time*math.cos(x[2])
    ret[1] = x[1] + u[0]*delta_time*math.sin(x[2])
    ret[2] = x[2] + u[1]*delta_time
    return ret

def control_model(time):
    u = numpy.zeros((2, 1))
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
    for i , rfid in enumerate(RFID):
        d = numpy.linalg.norm(rfid - x[:2]) # x[:2] means x[0], x[1]
        dz = y[i] - d
        w *= gaussian(dz, 0, 1.0)
    return w

def gaussian(x, u, sigma):
    return 1.0/math.sqrt(2.0*math.pi*sigma**2.0)*math.exp(-(x-u)**2.0/(2.0*sigma**2.0))


class Particle:
    """A simple Particle class"""

    def __init__(self, dimention):
        self.state = numpy.zeros((dimention, 1))
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
        self.estimation = None

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
        process_noise = numpy.random.normal(size=(self.num_of_particles, self.dimention, 1))
        for i, particle in enumerate(self.particles):
            self.particles[i].state = process_function(x=particle.state,
                                                       u=control, delta_time=dt)
            self.particles[i].state += (self.process_cov.dot(process_noise[i]))

    def sampling(self, observation, observe_function):
        for i, particle in enumerate(self.particles):
            self.particles[i].weight = observe_function(x=particle.state, y=observation)
        self.normalize()
        self.resampling()
        self.estimate()

    def normalize(self):
        sum_weight = sum([particle.weight for particle in self.particles])
        if sum_weight != 0:
            for i, particle in enumerate(self.particles):
                self.particles[i].weight = self.particles[i].weight / sum_weight
        else:
            for i, particle in enumerate(self.particles):
                self.particles[i].weight = 1.0 / float(self.num_of_particles)

    def resampling(self):
        pass

    def estimate(self):
        self.estimation = numpy.zeros((self.dimention, 1))
        for particle in (self.particles):
            self.estimation += particle.state * particle.weight
            


if __name__ == "__main__":
    particlefilter = ParticleFilter(dimention=STATE_DIMENTION, num_of_particles=100)
    process_mean = numpy.array([0, 0, 0])
    process_cov = numpy.array([[0.2, 0, 0], [0, 0.2, 0], [0, 0, math.radians(0.1)]])**2
    particlefilter.set_process_noise_param(mean=process_mean, cov=process_cov)
    observe_mean = numpy.array([0])
    observe_cov = numpy.array([1])
    particlefilter.set_observe_noise_param(mean=observe_mean, cov=observe_cov)
    particlefilter.print_info()
    # Simulation parameter
    simulation_cov = numpy.array([[0.2, 0], [0, math.radians(0.1)]])

    truth_state = numpy.zeros((3, 1))
    deadr_state = numpy.zeros((3, 1))
    control = numpy.zeros((1, 2))
    observation = numpy.zeros((len(RFID), 1))
    dt = 0.5
    time = 0.0
    truth_trajectory = []
    deadr_trajectory = []
    num_of_loop = 200
    pbar = ProgressBar(widgets=[Percentage(), Bar()], max_value=num_of_loop)
    for i in range(num_of_loop):
        # calculation ground truth
        time = time + dt
        control = control_model(time)
        truth_state = process_model(x=truth_state, u=control, delta_time=dt)
        deadr_state = process_model(x=deadr_state,
                                    u=(control
                                       +simulation_cov.dot(numpy.random.randn(2,1))),
                                    delta_time=dt)
        particlefilter.predict(dt, process_model, control)
        for ind , rfid in enumerate(RFID):
            observation[ind] = (numpy.linalg.norm(RFID[ind] - truth_state[:2])
                                + numpy.random.normal(observe_mean[0], observe_cov[0]))
        particlefilter.sampling(observation, observe_model)
        truth_trajectory.append(truth_state)
        deadr_trajectory.append(deadr_state)
        pbar.update(i+1)

    pbar.finish()
    plt.figure(facecolor="w")
    plt.scatter([x[0] for x in truth_trajectory], [x[1] for x in truth_trajectory],
                color='r', marker='x')
    plt.scatter([x[0] for x in deadr_trajectory], [x[1] for x in deadr_trajectory],
                color='b', marker='x')
    plt.axis([-30, 30.0, 0.0, 30.0])
    plt.show()
