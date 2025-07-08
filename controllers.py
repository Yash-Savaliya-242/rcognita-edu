"""
Contains controllers a.k.a. agents.

"""

from utilities import dss_sim
from utilities import rep_mat
from utilities import uptria2vec
from utilities import push_vec
import models
import numpy as np
import scipy as sp
from scipy.linalg import solve_discrete_are
from numpy.random import rand
from scipy.optimize import minimize
from scipy.optimize import basinhopping
from scipy.optimize import NonlinearConstraint
from scipy.stats import multivariate_normal
from numpy.linalg import lstsq 
from numpy import reshape
import warnings
import math
# For debugging purposes
from tabulate import tabulate
import os

def ctrl_selector(t, observation, action_manual, ctrl_nominal, ctrl_benchmarking, ctrl_lqr, mode):
    """
    Main interface for various controllers.

    Parameters
    ----------
    mode : : string
        Controller mode as acronym of the respective control method.

    Returns
    -------
    action : : array of shape ``[dim_input, ]``.
        Control action.

    """
    
    if mode=='manual': 
        action = action_manual
    elif mode=='Nominal': 
        action = ctrl_nominal.compute_action(t, observation)
    elif mode=='lqr':
        action=ctrl_lqr.compute_action(t,observation)
    else: # Controller for benchmakring
        action = ctrl_benchmarking.compute_action(t, observation)
        
    return action


class ControllerOptimalPredictive:
    """
    Class of predictive optimal controllers, primarily model-predictive control and predictive reinforcement learning, that optimize a finite-horizon cost.
    
    Currently, the actor model is trivial: an action is generated directly without additional policy parameters.
        
    Attributes
    ----------
    dim_input, dim_output : : integer
        Dimension of input and output which should comply with the system-to-be-controlled.
    mode : : string
        Controller mode. Currently available (:math:`\\rho` is the running objective, :math:`\\gamma` is the discounting factor):
          
        .. list-table:: Controller modes
           :widths: 75 25
           :header-rows: 1
    
           * - Mode
             - Cost function
           * - 'MPC' - Model-predictive control (MPC)
             - :math:`J_a \\left( y_1, \\{action\\}_1^{N_a} \\right)= \\sum_{k=1}^{N_a} \\gamma^{k-1} \\rho(y_k, u_k)`
           * - 'RQL' - RL/ADP via :math:`N_a-1` roll-outs of :math:`\\rho`
             - :math:`J_a \\left( y_1, \\{action\}_{1}^{N_a}\\right) = \\sum_{k=1}^{N_a-1} \\gamma^{k-1} \\rho(y_k, u_k) + \\hat Q^{\\theta}(y_{N_a}, u_{N_a})` 
           * - 'SQL' - RL/ADP via stacked Q-learning
             - :math:`J_a \\left( y_1, \\{action\\}_1^{N_a} \\right) = \\sum_{k=1}^{N_a-1} \\hat \\gamma^{k-1} Q^{\\theta}(y_{N_a}, u_{N_a})`               
        
        Here, :math:`\\theta` are the critic parameters (neural network weights, say) and :math:`y_1` is the current observation.
        
        *Add your specification into the table when customizing the agent*.    

    ctrl_bnds : : array of shape ``[dim_input, 2]``
        Box control constraints.
        First element in each row is the lower bound, the second - the upper bound.
        If empty, control is unconstrained (default).
    action_init : : array of shape ``[dim_input, ]``   
        Initial action to initialize optimizers.          
    t0 : : number
        Initial value of the controller's internal clock.
    sampling_time : : number
        Controller's sampling time (in seconds).
    Nactor : : natural number
        Size of prediction horizon :math:`N_a`. 
    pred_step_size : : number
        Prediction step size in :math:`J_a` as defined above (in seconds). Should be a multiple of ``sampling_time``. Commonly, equals it, but here left adjustable for
        convenience. Larger prediction step size leads to longer factual horizon.
    sys_rhs, sys_out : : functions        
        Functions that represent the right-hand side, resp., the output of the exogenously passed model.
        The latter could be, for instance, the true model of the system.
        In turn, ``state_sys`` represents the (true) current state of the system and should be updated accordingly.
        Parameters ``sys_rhs, sys_out, state_sys`` are used in those controller modes which rely on them.
    buffer_size : : natural number
        Size of the buffer to store data.
    gamma : : number in (0, 1]
        Discounting factor.
        Characterizes fading of running objectives along horizon.
    Ncritic : : natural number
        Critic stack size :math:`N_c`. The critic optimizes the temporal error which is a measure of critic's ability to capture the
        optimal infinite-horizon cost (a.k.a. the value function). The temporal errors are stacked up using the said buffer.
    critic_period : : number
        The critic is updated every ``critic_period`` units of time. 
    critic_struct : : natural number
        Choice of the structure of the critic's features.
        
        Currently available:
            
        .. list-table:: Critic structures
           :widths: 10 90
           :header-rows: 1
    
           * - Mode
             - Structure
           * - 'quad-lin'
             - Quadratic-linear
           * - 'quadratic'
             - Quadratic
           * - 'quad-nomix'
             - Quadratic, no mixed terms
           * - 'quad-mix'
             - Quadratic, no mixed terms in input and output, i.e., :math:`w_1 y_1^2 + \\dots w_p y_p^2 + w_{p+1} y_1 u_1 + \\dots w_{\\bullet} u_1^2 + \\dots`, 
               where :math:`w` is the critic's weight vector
       
        *Add your specification into the table when customizing the critic*. 
    run_obj_struct : : string
        Choice of the running objective structure.
        
        Currently available:
           
        .. list-table:: Critic structures
           :widths: 10 90
           :header-rows: 1
    
           * - Mode
             - Structure
           * - 'quadratic'
             - Quadratic :math:`\\chi^\\top R_1 \\chi`, where :math:`\\chi = [observation, action]`, ``run_obj_pars`` should be ``[R1]``
           * - 'biquadratic'
             - 4th order :math:`\\left( \\chi^\\top \\right)^2 R_2 \\left( \\chi \\right)^2 + \\chi^\\top R_1 \\chi`, where :math:`\\chi = [observation, action]`, ``run_obj_pars``
               should be ``[R1, R2]``   
        
        *Pass correct run objective parameters in* ``run_obj_pars`` *(as a list)*
        
        *When customizing the running objective, add your specification into the table above*
        
    References
    ----------
    .. [1] Osinenko, Pavel, et al. "Stacked adaptive dynamic programming with unknown system model." IFAC-PapersOnLine 50.1 (2017): 4150-4155        
        
    """       
    def __init__(self,
                 dim_input,
                 dim_output,
                 mode='MPC',
                 ctrl_bnds=[],
                 action_init = [],
                 t0=0,
                 sampling_time=0.1,
                 Nactor=1,
                 pred_step_size=0.1,
                 sys_rhs=[],
                 sys_out=[],
                 state_sys=[],
                 buffer_size=20,
                 gamma=1,
                 Ncritic=4,
                 critic_period=0.1,
                 critic_struct='quad-nomix',
                 run_obj_struct='quadratic',
                 run_obj_pars=[],
                 observation_target=[],
                 state_init=[],
                 obstacle=[],
                 seed=1):
        """
            Parameters
            ----------
            dim_input, dim_output : : integer
                Dimension of input and output which should comply with the system-to-be-controlled.
            mode : : string
                Controller mode. Currently available (:math:`\\rho` is the running objective, :math:`\\gamma` is the discounting factor):
                
                .. list-table:: Controller modes
                :widths: 75 25
                :header-rows: 1
            
                * - Mode
                    - Cost function
                * - 'MPC' - Model-predictive control (MPC)
                    - :math:`J_a \\left( y_1, \\{action\\}_1^{N_a} \\right)= \\sum_{k=1}^{N_a} \\gamma^{k-1} \\rho(y_k, u_k)`
                * - 'RQL' - RL/ADP via :math:`N_a-1` roll-outs of :math:`\\rho`
                    - :math:`J_a \\left( y_1, \\{action\}_{1}^{N_a}\\right) = \\sum_{k=1}^{N_a-1} \\gamma^{k-1} \\rho(y_k, u_k) + \\hat Q^{\\theta}(y_{N_a}, u_{N_a})` 
                * - 'SQL' - RL/ADP via stacked Q-learning
                    - :math:`J_a \\left( y_1, \\{action\\}_1^{N_a} \\right) = \\sum_{k=1}^{N_a-1} \\gamma^{k-1} \\hat Q^{\\theta}(y_{N_a}, u_{N_a})`               
                
                Here, :math:`\\theta` are the critic parameters (neural network weights, say) and :math:`y_1` is the current observation.
                
                *Add your specification into the table when customizing the agent* .   
        
            ctrl_bnds : : array of shape ``[dim_input, 2]``
                Box control constraints.
                First element in each row is the lower bound, the second - the upper bound.
                If empty, control is unconstrained (default).
            action_init : : array of shape ``[dim_input, ]``   
                Initial action to initialize optimizers.              
            t0 : : number
                Initial value of the controller's internal clock
            sampling_time : : number
                Controller's sampling time (in seconds)
            Nactor : : natural number
                Size of prediction horizon :math:`N_a` 
            pred_step_size : : number
                Prediction step size in :math:`J` as defined above (in seconds). Should be a multiple of ``sampling_time``. Commonly, equals it, but here left adjustable for
                convenience. Larger prediction step size leads to longer factual horizon.
            sys_rhs, sys_out : : functions        
                Functions that represent the right-hand side, resp., the output of the exogenously passed model.
                The latter could be, for instance, the true model of the system.
                In turn, ``state_sys`` represents the (true) current state of the system and should be updated accordingly.
                Parameters ``sys_rhs, sys_out, state_sys`` are used in those controller modes which rely on them.
            buffer_size : : natural number
                Size of the buffer to store data.
            gamma : : number in (0, 1]
                Discounting factor.
                Characterizes fading of running objectives along horizon.
            Ncritic : : natural number
                Critic stack size :math:`N_c`. The critic optimizes the temporal error which is a measure of critic's ability to capture the
                optimal infinite-horizon cost (a.k.a. the value function). The temporal errors are stacked up using the said buffer.
            critic_period : : number
                The critic is updated every ``critic_period`` units of time. 
            critic_struct : : natural number
                Choice of the structure of the critic's features.
                
                Currently available:
                    
                .. list-table:: Critic feature structures
                :widths: 10 90
                :header-rows: 1
            
                * - Mode
                    - Structure
                * - 'quad-lin'
                    - Quadratic-linear
                * - 'quadratic'
                    - Quadratic
                * - 'quad-nomix'
                    - Quadratic, no mixed terms
                * - 'quad-mix'
                    - Quadratic, no mixed terms in input and output, i.e., :math:`w_1 y_1^2 + \\dots w_p y_p^2 + w_{p+1} y_1 u_1 + \\dots w_{\\bullet} u_1^2 + \\dots`, 
                    where :math:`w` is the critic's weights
            
                *Add your specification into the table when customizing the critic*.
            run_obj_struct : : string
                Choice of the running objective structure.
                
                Currently available:
                
                .. list-table:: Running objective structures
                :widths: 10 90
                :header-rows: 1
            
                * - Mode
                    - Structure
                * - 'quadratic'
                    - Quadratic :math:`\\chi^\\top R_1 \\chi`, where :math:`\\chi = [observation, action]`, ``run_obj_pars`` should be ``[R1]``
                * - 'biquadratic'
                    - 4th order :math:`\\left( \\chi^\\top \\right)^2 R_2 \\left( \\chi \\right)^2 + \\chi^\\top R_1 \\chi`, where :math:`\\chi = [observation, action]`, ``run_obj_pars``
                    should be ``[R1, R2]``
            """        

        np.random.seed(seed)
        # print(seed)

        self.dim_input = dim_input
        self.dim_output = dim_output
        
        self.mode = mode

        self.ctrl_clock = t0
        self.sampling_time = sampling_time
        
        # Controller: common
        self.Nactor = Nactor 
        self.pred_step_size = pred_step_size
        
        self.action_min = np.array( ctrl_bnds[:,0] )
        self.action_max = np.array( ctrl_bnds[:,1] )
        self.action_sqn_min = rep_mat(self.action_min, 1, Nactor)
        self.action_sqn_max = rep_mat(self.action_max, 1, Nactor) 
        self.action_sqn_init = []
        self.state_init = []

        if len(action_init) == 0:
            self.action_curr = self.action_min/10
            self.action_sqn_init = rep_mat( self.action_min/10 , 1, self.Nactor)
            self.action_init = self.action_min/10
        else:
            self.action_curr = action_init
            self.action_sqn_init = rep_mat( action_init , 1, self.Nactor)
        
        
        self.action_buffer = np.zeros( [buffer_size, dim_input] )
        self.observation_buffer = np.zeros( [buffer_size, dim_output] )    
        self.t=0    
        
        # Exogeneous model's things
        self.sys_rhs = sys_rhs
        self.sys_out = sys_out
        self.state_sys = state_sys   
        
        # Learning-related things
        self.buffer_size = buffer_size
        self.critic_clock = t0
        self.gamma = gamma
        self.Ncritic = Ncritic
        self.Ncritic = np.min([self.Ncritic, self.buffer_size-1]) # Clip critic buffer size
        self.critic_period = critic_period
        self.critic_struct = critic_struct
        self.run_obj_struct = run_obj_struct
        self.run_obj_pars = run_obj_pars
        self.observation_target = observation_target
        
        self.accum_obj_val = 0
        # print('---Critic structure---', self.critic_struct)

        if self.critic_struct == 'quad-lin':
            self.dim_critic = int( ( ( self.dim_output + self.dim_input ) + 1 ) * ( self.dim_output + self.dim_input )/2 + (self.dim_output + self.dim_input) ) 
            self.Wmin = -1e3*np.ones(self.dim_critic) 
            self.Wmax = 1e3*np.ones(self.dim_critic) 
        elif self.critic_struct == 'quadratic':
            self.dim_critic = int( ( ( self.dim_output + self.dim_input ) + 1 ) * ( self.dim_output + self.dim_input )/2 )
            self.Wmin = np.zeros(self.dim_critic) 
            self.Wmax = 1e3*np.ones(self.dim_critic)    
        elif self.critic_struct == 'quad-nomix':
            self.dim_critic = self.dim_output + self.dim_input
            self.Wmin = np.zeros(self.dim_critic) 
            self.Wmax = 1e3*np.ones(self.dim_critic)    
        elif self.critic_struct == 'quad-mix':
            self.dim_critic = int( self.dim_output + self.dim_output * self.dim_input + self.dim_input )
            self.Wmin = -1e3*np.ones(self.dim_critic)  
            self.Wmax = 1e3*np.ones(self.dim_critic) 
        elif self.critic_struct == 'poly3':
            self.dim_critic = int( ( ( self.dim_output + self.dim_input ) + 1 ) * ( self.dim_output + self.dim_input ) )
            self.Wmin = -1e3*np.ones(self.dim_critic)  
            self.Wmax = 1e3*np.ones(self.dim_critic) 
        elif self.critic_struct == 'poly4':
            self.dim_critic = int( ( ( self.dim_output + self.dim_input ) + 1 ) * ( self.dim_output + self.dim_input )/2 * 3)
            self.Wmin = np.zeros(self.dim_critic) 
            self.Wmax = np.ones(self.dim_critic) 
        self.N_CTRL = N_CTRL(k_rho=1.0, k_alpha=4.0, k_beta=-1.5, # Example gains, vary for Experiment A
        target_x=0.0, target_y=0.0, target_theta=0.0,
        sampling_time=sampling_time,
        dim_input=dim_input, dim_output=dim_output)
    

    def reset(self,t0):
        """
        Resets agent for use in multi-episode simulation.
        Only internal clock, value and current actions are reset.
        All the learned parameters are retained.
        
        """        

        # Controller: common

        if len(self.action_init) == 0:
            self.action_curr = self.action_min/10
            self.action_sqn_init = rep_mat( self.action_min/10 , 1, self.Nactor)
            self.action_init = self.action_min/10
        else:
            self.action_curr = self.action_init
            self.action_sqn_init = rep_mat( self.action_init , 1, self.Nactor)
        
        self.action_buffer = np.zeros( [self.buffer_size, self.dim_input] )
        self.observation_buffer = np.zeros( [self.buffer_size, self.dim_output] )        

        self.critic_clock = t0
        self.ctrl_clock = t0
    
    def receive_sys_state(self, state):
        """
        Fetch exogenous model state. Used in some controller modes. See class documentation.

        """
        self.state_sys = state
    
    def upd_accum_obj(self, observation, action):
        """
        Sample-to-sample accumulated (summed up or integrated) running objective. This can be handy to evaluate the performance of the agent.
        If the agent succeeded to stabilize the system, ``accum_obj`` would converge to a finite value which is the performance mark.
        The smaller, the better (depends on the problem specification of course - you might want to maximize cost instead).
        
        """
        self.accum_obj_val += self.run_obj(observation, action)*self.sampling_time
                 
    def run_obj(self, observation, action,terminal=False):
        """
        Running (equivalently, instantaneous or stage) objective. Depending on the context, it is also called utility, reward, running cost etc.
        
        See class documentation.
        """
        observation_arr = np.array(observation)
        action_arr = np.array(action)

        # Handle observation target: calculate error if a target is provided
        self.observation_target=np.array([0.0,0.0,0.0])
        observation_err = observation_arr - self.observation_target
        
        # Form the combined state-action vector chi
        chi = np.concatenate((observation_err, action_arr))
        run_obj=0

        

        # if self.t==self.Nactor-1:
        #     terminal=True

        if self.run_obj_struct == 'quadratic':
            # run_obj_pars should be [R1] where R1 is a matrix for chi.T @ R1 @ chi
            if not self.run_obj_pars or len(self.run_obj_pars) < 1:
                warnings.warn("run_obj_pars is empty or malformed for 'quadratic' mode. Returning 0 for running objective.")
                return 0.0 # Return a float
            R1 = np.array(self.run_obj_pars[0])
            # print(R1)
            # Check dimensions for compatibility: R1 must be square and match chi's dimension
            expected_dim = chi.shape[0]
            if R1.shape != (expected_dim, expected_dim):
                warnings.warn(f"R1 matrix dimension mismatch. Expected {expected_dim}x{expected_dim}, got {R1.shape}. Returning 0 for running objective.")
                return 0.0 # Return a float
            run_obj = chi.T @ R1 @ chi
            self.t+=1
            # print(self.t)

            # Add terminal cost if requested
            if terminal:
                if len(self.run_obj_pars) < 2:
                    warnings.warn("Qf not provided in run_obj_pars for terminal cost. Skipping terminal cost.")
                else:
                    Qf = np.array(self.run_obj_pars[1])
                    obs_dim = observation_err.shape[0]
                    if Qf.shape != (obs_dim, obs_dim):
                        warnings.warn(f"Qf matrix dimension mismatch. Expected {obs_dim}x{obs_dim}, got {Qf.shape}. Skipping terminal cost.")
                    else:
                        run_obj += observation_err.T @ Qf @ observation_err
                    
                    

        return run_obj

    def _actor_cost(self, action_sqn, observation):
        """
        See class documentation.
        
        Customization
        -------------        
        
        Introduce your mode and the respective actor loss in this method. Don't forget to provide description in the class documentation.

        """
        
        my_action_sqn = np.reshape(action_sqn, [self.Nactor, self.dim_input])
        
        observation_sqn = np.zeros([self.Nactor, self.dim_output])
        
        # System observation prediction
        observation_sqn[0, :] = observation
        state = self.state_sys
        for k in range(1, self.Nactor):
            state = state + self.pred_step_size * self.sys_rhs(0.01, state, my_action_sqn[k-1, :])  # Euler scheme
            
            observation_sqn[k, :] = self.sys_out(state)
        
        J = 0         
        if self.mode=='MPC':
            for k in range(self.Nactor):
                
                if k==self.Nactor-1:
                    J+=self.gamma**k * self.run_obj(observation_sqn[k, :], my_action_sqn[k, :],terminal=True)

                J += self.gamma**k * self.run_obj(observation_sqn[k, :], my_action_sqn[k, :])

        return J
    
    def _actor_optimizer(self, observation):
        """
        This method is merely a wrapper for an optimizer that minimizes :func:`~controllers.ControllerOptimalPredictive._actor_cost`.
        See class documentation.
        
        Customization
        -------------         
        
        This method normally should not be altered, adjust :func:`~controllers.ControllerOptimalPredictive._actor_cost` instead.
        The only customization you might want here is regarding the optimization algorithm.

        

        # For direct implementation of state constraints, this needs `partial` from `functools`
        # See [here](https://stackoverflow.com/questions/27659235/adding-multiple-constraints-to-scipy-minimize-autogenerate-constraint-dictionar)
        # def state_constraint(action_sqn, idx):
            
        #     my_action_sqn = np.reshape(action_sqn, [N, self.dim_input])
            
        #     observation_sqn = np.zeros([idx, self.dim_output])    
            
        #     # System output prediction
        #     if (mode==1) or (mode==3) or (mode==5):    # Via exogenously passed model
        #         observation_sqn[0, :] = observation
        #         state = self.state_sys
        #         Y[0, :] = observation
        #         x = self.x_s
        #         for k in range(1, idx):
        #             # state = get_next_state(state, my_action_sqn[k-1, :], delta)
        #             state = state + delta * self.sys_rhs([], state, my_action_sqn[k-1, :], [])  # Euler scheme
        #             observation_sqn[k, :] = self.sys_out(state)            
            
        #     return observation_sqn[-1, 1] - 1

        # my_constraints=[]
        # for my_idx in range(1, self.Nactor+1):
        #     my_constraints.append({'type': 'eq', 'fun': lambda action_sqn: state_constraint(action_sqn, idx=my_idx)})

        # my_constraints = {'type': 'ineq', 'fun': state_constraint}

        # Optimization method of actor    
        # Methods that respect constraints: BFGS, L-BFGS-B, SLSQP, trust-constr, Powell
        # actor_opt_method = 'SLSQP' # Standard
        """
        
        actor_opt_method = 'SLSQP'
        if actor_opt_method == 'trust-constr':
            actor_opt_options = {'maxiter': 40, 'disp': False} #'disp': True, 'verbose': 2}
        else:
            actor_opt_options = {'maxiter': 40, 'maxfev': 60, 'disp': False, 'adaptive': True, 'xatol': 1e-3, 'fatol': 1e-3}
       
        isGlobOpt = 0
        
        my_action_sqn_init = np.reshape(self.action_sqn_init, [self.Nactor*self.dim_input,])
        
        bnds = sp.optimize.Bounds(self.action_sqn_min, self.action_sqn_max, keep_feasible=True)
        
        try:
            if isGlobOpt:
                minimizer_kwargs = {'method': actor_opt_method, 'bounds': bnds, 'tol': 1e-3, 'options': actor_opt_options}
                action_sqn = basinhopping(lambda action_sqn: self._actor_cost(action_sqn, observation),
                                          my_action_sqn_init,
                                          minimizer_kwargs=minimizer_kwargs,
                                          niter = 10).x
            else:
                action_sqn = minimize(lambda action_sqn: self._actor_cost(action_sqn, observation),
                                      my_action_sqn_init,
                                      method=actor_opt_method,
                                      tol=1e-3,
                                      bounds=bnds,
                                      options=actor_opt_options).x        

        except ValueError:
            print('Actor''s optimizer failed. Returning default action')
            action_sqn = self.action_curr

        ##print(action_sqn)
        
        return action_sqn[:self.dim_input]    # Return first action
                    
    def compute_action(self, t, observation):
        """
        Main method. See class documentation.
        
        Customization
        -------------         
        
        Add your modes, that you introduced in :func:`~controllers.ControllerOptimalPredictive._actor_cost`, here.

        """       
        
        time_in_sample = t - self.ctrl_clock
        
        if time_in_sample >= self.sampling_time: # New sample
            # Update controller's internal clock
            self.ctrl_clock = t

            action = None
            
            if self.mode == 'MPC':  
                
                action = self._actor_optimizer(observation)
                # print(action.shape)

            elif self.mode == "Nominal":
                
                action = self.N_CTRL.compute_action(observation)
            if action is not None:
               self.action_curr = action
        
            
            return action    
    
        else:
            return self.action_curr



# New/Updated N_CTRL Class
class N_CTRL:
    def __init__(self, k_rho=2.5, k_alpha=-3.5, k_beta=2.5,
                 target_x=0.0, target_y=0.0, target_theta=0.0,
                 sampling_time=0.01, dim_input=2, dim_output=3): # Added dim_input/output for consistency
        self.k_rho = k_rho
        self.k_alpha = k_alpha
        self.k_beta = k_beta
        self.dt = sampling_time # Use sampling_time for internal dt consistency

        self.target_x = target_x
        self.target_y = target_y
        self.target_theta = target_theta

        self.ctrl_clock = 0.0 # Initial time for the controller's internal clock
        self.sampling_time = sampling_time
        self.action_curr = np.array([0.0, 0.0]) # Initialize current action to zero
        self.accum_obj_val = 0.0 # Initialize accumulated objective value
        self.dim_input = dim_input
        self.dim_output = dim_output

        # Dummy attributes for compatibility with logger/visualizer
        self.state_sys = np.zeros(dim_output) # N_CTRL doesn't use this directly, but visualizer might expect it.
                                             # It will be updated by receive_sys_state if called.

    def wrap_angle(self, angle):
        while angle > math.pi:
            angle -= 2 * math.pi
        while angle <= -math.pi:
            angle += 2 * math.pi
        return angle

    def compute_action(self, t, observation):
        """
        Computes the control action for the 3-wheel robot.
        """
        time_in_sample = t - self.ctrl_clock

        if t >= self.ctrl_clock + self.sampling_time - 1e-6:

            self.ctrl_clock = t # Update controller's internal clock

            # print(observation)

            x, y, th = observation
            x_f = self.target_x
            y_f = self.target_y
            th_f = self.target_theta

            dx = x_f - x
            dy = y_f - y
            rho = math.sqrt(dx**2 + dy**2)

            alpha = self.wrap_angle(-th + math.atan2(dy, dx))
            beta = self.wrap_angle((th_f - th) - alpha)

            # Debug prints - these are very useful!
            print(f"t: {t:.2f}, Obs: ({x:.2f}, {y:.2f}, {th:.2f})")
            print(f"Target: ({x_f:.2f}, {y_f:.2f}, {th_f:.2f})")
            # print(f"Errors: rho={rho:.4f}, alpha={alpha:.4f}, beta={beta:.4f}")

            # Define a small tolerance for "at target"
            pos_tolerance = 0.05 # meters
            angle_tolerance = 0.05 # radians

            if rho < pos_tolerance:
                if abs(self.wrap_angle(th_f - th)) < angle_tolerance:
                    v = 0.0
                    w = 0.0
                # print("--> At target, stopping. v=0, w=0")
            else:
                v = self.k_rho * rho
                w = self.k_alpha * alpha + self.k_beta * beta
                # print(f"--> Control action: v={v:.4f}, w={w:.4f}")

            v = np.clip(v, -1, 1)
            w = np.clip(w, -1.0, 1.0)


            self.action_curr = np.array([v, w])
            return self.action_curr
        else:
            return self.action_curr # Return the last computed action if not a new sample

    def reset(self, t0):
        """
        Resets controller for new episode.
        """
        self.ctrl_clock = t0
        self.action_curr = np.array([0.0, 0.0])
        self.accum_obj_val = 0.0 # Reset accumulated objective

    def receive_sys_state(self, state):
        """
        Placeholder method for compatibility with the simulation framework.
        N_CTRL doesn't actively use the full system state for its internal logic,
        but the main simulation loop might try to pass it.
        """
        self.state_sys = state # Store it just in case, or do nothing if truly not needed.

    def upd_accum_obj(self, observation, action):
        """
        Accumulates the running objective for performance evaluation.
        """
        self.accum_obj_val += self.run_obj(observation, action) * self.sampling_time

    def run_obj(self, observation, action):
        """
        Calculates a running objective (cost) for the N_CTRL.
        This is primarily for logging and visualization, not for the controller's internal logic.
        """
        x, y, th = observation
        x_f = self.target_x
        y_f = self.target_y
        th_f = self.target_theta

        # Cost components: position error squared and orientation error squared, and control effort.
        pos_cost = (x - x_f)**2 + (y - y_f)**2
        # Use a proper angle difference for orientation cost
        orientation_cost = self.wrap_angle(th - th_f)**2

        # Control effort cost (e.g., squared velocities)
        v, w = action
        control_effort_cost = 0.1 * (v**2 + w**2) # Small weight on control effort

        # You can tune these weights
        total_cost = pos_cost * 10 + orientation_cost * 5 + control_effort_cost
        return total_cost
    
class LQR:
    def __init__(self,A,B,Q,R,obs_target=[0.0,0.0,0.0],sampling_time=0.01,):
        self.A=A
        self.B=B
        self.Q=Q 
        self.R=R
        self.observation_target=obs_target
        self.ctrl_clock = 0.0 
        self.sampling_time =sampling_time
        self.action_curr = np.array([0.0, 0.0]) # Initialize current action to zero
        self.accum_obj_val = 0.0

    def compute_gain(self):

        # Solve the Discrete Algebraic Riccati Equation (DARE)
        P = solve_discrete_are(self.A, self.B, self.Q, self.R)

        K=np.linalg.inv(self.R+self.B.T@P@self.B)@self.B.T@P@self.A

        return K

    def compute_action(self,t,observation):
        # time_in_sample = t - self.ctrl_clock

        if t >= self.ctrl_clock + self.sampling_time - 1e-6:

            self.ctrl_clock = t # Update controller's internal clock

            # print(observation)
            # observation_target=[]

            state_err = observation - self.observation_target
            # print(state_err.shape)

            self.K=self.compute_gain()


            u=-self.K@state_err
            # print(u)

            # x=self.A@state_err +self.B@u

            self.action_curr = u

            # print(x)
            return self.action_curr
        else:
            return self.action_curr #
        

    def reset(self, t0):
        """
        Resets controller for new episode.
        """
        self.ctrl_clock = t0
        self.action_curr = np.array([0.0, 0.0])
        self.accum_obj_val = 0.0 # Reset accumulated objective

    def receive_sys_state(self, state):
        """
        Placeholder method for compatibility with the simulation framework.
        N_CTRL doesn't actively use the full system state for its internal logic,
        but the main simulation loop might try to pass it.
        """
        self.state_sys = state # Store it just in case, or do nothing if truly not needed.

    def upd_accum_obj(self, observation, action):
        """
        Accumulates the running objective for performance evaluation.
        """
        self.accum_obj_val += self.run_obj(observation, action) * self.sampling_time

    def run_obj(self, observation, action):
        """
        Calculates a running objective (cost) for the N_CTRL.
        This is primarily for logging and visualization, not for the controller's internal logic.
        """
        state_error = np.diag(observation - self.observation_target)
        action=np.diag(action)
        running_cost = state_error.T @ self.Q @ state_error 
        # action.T @ self.R@ action

        # print(running_cost.shape)
        return np.linalg.det(running_cost)
    
    
        




