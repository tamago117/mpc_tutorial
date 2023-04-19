# reference https://gist.github.com/mayataka/7e608dbbbcd93d232cf44e6cab6b0332

from casadi import *
import math
import numpy as np

from CartPole import CartPole

class cartpole_MPC:
    def __init__(self, T, N, max_input, x_max):
        self.T = T # horizon length
        self.N = N # discreate grid number
        self.dt = self.T/self.N # minute time
        self.nx = 4 # state variable number
        self.nu = 1 # input variable number
        self.cartpole = CartPole()

        self.Q  = [1.0, 2.0, 0.05, 0.05]       # state weights
        self.Qf = [1.0, 2.0, 0.1, 0.1]       # terminal state weights
        self.R  = [0.01]                       # input weights

        self.max_input = max_input
        self.x_max = x_max

        w = [] # contain optimal variable
        w0 = [] # contain initial optimal variable
        lbw = [] # lower bound optimal variable
        ubw = [] # upper bound optimal variable
        J = 0 # cost function
        g  = [] # constrain
        lbg = [] # lower bound constrain
        ubg = [] # upper bound constrain
        lam_x0 = [] # Lagrangian multiplier
        lam_g0 = [] # Lagrangian multiplier

        Xk = MX.sym('X0', self.nx) # initial time state vector x0
        Xref = MX.sym('x_ref', self.nx) # x reference

        w += [Xk]
        # equality constraint
        lbw += [0, 0, 0, 0]  # constraints are set by setting lower-bound and upper-bound to the same value
        ubw += [0, 0, 0, 0]      # constraints are set by setting lower-bound and upper-bound to the same value
        w0 +=  [0, 0, 0, 0]      # x0 initial estimate
        lam_x0 += [0, 0, 0, 0]    # Lagrangian multiplier initial estimate

        for k in range(self.N):
            Uk = MX.sym('U_' + str(k), self.nu)
            w += [Uk]
            lbw += [-self.max_input]
            ubw += [self.max_input]
            w0 += [0]
            lam_x0 += [0]

            #stage cost
            J += self.stage_cost(Xk, Uk, Xref)

            # Discretized equation of state by forward Euler
            dXk = self.cartpole.dynamics(Xk, Uk)
            Xk_next = vertcat(Xk[0] + dXk[0] * self.dt,
                              Xk[1] + dXk[1] * self.dt,
                              Xk[2] + dXk[2] * self.dt,
                              Xk[3] + dXk[3] * self.dt)
            Xk1 = MX.sym('X_' + str(k+1), self.nx)
            w   += [Xk1]
            lbw += [-self.x_max, -inf, -inf, -inf]
            ubw += [self.x_max, inf, inf, inf]
            w0 += [0.0, 0.0, 0.0, 0.0]
            lam_x0 += [0, 0, 0, 0]

            # (xk+1=xk+fk*dt) is introduced as an equality constraint
            g   += [Xk_next-Xk1]
            lbg += [0, 0, 0, 0]     # Equality constraints are set by setting lower-bound and upper-bound to the same value
            ubg += [0, 0, 0, 0]     # Equality constraints are set by setting lower-bound and upper-bound to the same value
            lam_g0 += [0, 0, 0, 0]
            Xk = Xk1

        # finite cost
        J += self.terminal_cost(Xk, Xref)

        self.J = J
        self.w = vertcat(*w)
        self.g = vertcat(*g)
        self.x = w0
        self.lam_x = lam_x0
        self.lam_g = lam_g0
        self.lbx = lbw
        self.ubx = ubw
        self.lbg = lbg
        self.ubg = ubg

        # 非線形計画問題(NLP)
        self.nlp = {'f': self.J, 'x': self.w, 'p': Xref, 'g': self.g}
        # Ipopt ソルバー，最小バリアパラメータを0.1，最大反復回数を5, ウォームスタートをONに
        self.solver = nlpsol('solver', 'ipopt', self.nlp, {'calc_lam_p':True, 'calc_lam_x':True, 'print_time':False, 'ipopt':{'max_iter':5, 'mu_min':0.1, 'warm_start_init_point':'yes', 'print_level':0, 'print_timing_statistics':'no'}})


    def init(self, x0=None, xref=None):
        if x0 is not None:
            # 初期状態についての制約を設定
            self.lbx[0:4] = x0
            self.ubx[0:4] = x0
        if xref is None:
            xref = np.zeros(self.nx)
        # primal variables (x) と dual variables（ラグランジュ乗数）の初期推定解も与えつつ solve（warm start）
        sol = self.solver(x0=self.x, p=xref, lbx=self.lbx, ubx=self.ubx, lbg=self.lbg, ubg=self.ubg, lam_x0=self.lam_x, lam_g0=self.lam_g)
        # 次の warm start のために解を保存
        self.x = sol['x'].full().flatten()
        self.lam_x = sol['lam_x'].full().flatten()
        self.lam_g = sol['lam_g'].full().flatten()


    def stage_cost(self, x, u, x_ref):
        cost = 0
        for i in range(self.nx):
            cost += 0.5 * self.Q[i] * (x[i] - x_ref[i])**2
        for i in range(self.nu):
            cost += 0.5 * self.R[i] * u[i]**2
        return cost


    def terminal_cost(self, x, x_ref):
        cost = 0
        for i in range(self.nx):
            cost += 0.5 * self.Q[i] * (x[i] - x_ref[i])**2
        return cost


    """
    x0 = np.array([x_current, angle_current, v_current, angleV_current])
    xref = np.array([x_ref, theta_ref, v_ref, angleV_ref])
    """
    def solve(self, x0, xref):
        # 初期状態についての制約を設定
        nx = x0.shape[0]
        self.lbx[0:nx] = x0
        self.ubx[0:nx] = x0
        # primal variables (x) と dual variables（ラグランジュ乗数）の初期推定解も与えつつ solve（warm start）
        sol = self.solver(x0=self.x, p=xref, lbx=self.lbx, ubx=self.ubx, lbg=self.lbg, ubg=self.ubg, lam_x0=self.lam_x, lam_g0=self.lam_g)
        # 次の warm start のために解を保存
        self.x = sol['x'].full().flatten()
        self.lam_x = sol['lam_x'].full().flatten()
        self.lam_g = sol['lam_g'].full().flatten()
        return np.array([self.x[4]]) # 制御入力を returns