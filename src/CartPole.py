# reference https://gist.github.com/mayataka/7e608dbbbcd93d232cf44e6cab6b0332

from casadi import *

class CartPole:
    def __init__(self):
        self.mc = 2.0 #cart mass
        self.mp = 0.2 #pole mass
        self.l = 0.5 #pole length
        self.ga = 9.81 #gravity constant

    def dynamics(self, x, u):
        y = x[0] #cart position[m]
        th = x[1] #pole angle[rad]
        dy = x[2] #cart velocity[m/s]
        dth = x[3] #pole angle velocity[rad/s]
        f = u[0] #input[N]

        #cart acceleration
        ddy = (f+self.mp*sin(th)*(self.l*dth*dth+self.ga*cos(th))) / (self.mc+self.mp*sin(th)*sin(th))
        # pole angle acceleration
        ddth = (-f*cos(th)-self.mp*self.l*dth*dth*cos(th)*sin(th)-(self.mc+self.mp)*self.ga*sin(th)) / (self.l * (self.mc+self.mp*sin(th)*sin(th)))
        
        return dy, dth, ddy, ddth