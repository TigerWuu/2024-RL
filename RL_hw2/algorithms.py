import numpy as np
import json
from collections import deque

from gridworld import GridWorld

# =========================== 2.1 model free prediction ===========================
class ModelFreePrediction:
    """
    Base class for ModelFreePrediction algorithms
    """
    def __init__(self, grid_world: GridWorld, policy: np.ndarray = None, discount_factor: float = 1.0, max_episode: int = 300, seed: int = 1):
        """
        Args:
            grid_world (GridWorld): GridWorld object
            policy (np.ndarray): Stochastic policy representing action probabilities [state_space, action_space]
            discount_factor (float, optional): Discount factor gamma. Defaults to 1.0.
            max_episdoe (int, optional): Maximum episdoe for data collection. Defaults to 10000.
            seed (int): seed for sampling action from the policy
        """
        self.grid_world = grid_world
        self.discount_factor = discount_factor
        self.max_episode = max_episode
        self.episode_counter = 0  
        self.action_space = grid_world.get_action_space()
        self.state_space  = grid_world.get_state_space()
        self.values       = np.zeros(self.state_space)
        self.rng = np.random.default_rng(seed)      # only call this in collect_data()
        if policy:
            self.policy = policy
        else:
            self.policy = np.ones((self.state_space, self.action_space)) / self.action_space  # random policy

    def get_all_state_values(self) -> np.array:
        return self.values

    def collect_data(self) -> tuple:
        """
        Use the stochastic policy to interact with the environment and collect one step of data.
        Samples an action based on the action probability distribution for the current state.
        """

        current_state = self.grid_world.get_current_state()  # Get the current state
        
        # Sample an action based on the stochastic policy's probabilities for the current state
        action_probs = self.policy[current_state]  
        action = self.rng.choice(self.action_space, p=action_probs)  

        next_state, reward, done = self.grid_world.step(action)  
        if done:
            self.episode_counter +=1
        return next_state, reward, done
        

class MonteCarloPrediction(ModelFreePrediction):
    def __init__(self, grid_world: GridWorld, policy: np.ndarray = None, discount_factor: float = 1.0, max_episode: int = 300, seed: int = 1):
        """
        Constructor for MonteCarloPrediction
        
        Args:
            grid_world (GridWorld): GridWorld object
            policy (np.ndarray): Stochastic policy representing action probabilities [state_space, action_space]
            discount_factor (float, optional): Discount factor gamma. Defaults to 1.0.
            max_episdoe (int, optional): Maximum episdoe for data collection. Defaults to 10000.
        """
        super().__init__(grid_world,policy, discount_factor, max_episode, seed)
        self.visit_states_count = np.zeros(self.state_space)
        self.Gt_states = np.zeros(self.state_space)

    def update_states_value(self, v_s, first_visited_key):
        for i in range(len(v_s)):
            if i in first_visited_key:
                self.values[v_s[i]] = self.values[v_s[i]] + (self.Gt_states[v_s[i]] - self.values[v_s[i]]) / self.visit_states_count[v_s[i]]
    
    def cal_Gt_states(self,visited_states, reward_traced, first_visited_key):
        for i in range(len(visited_states)):
            if i in first_visited_key:
                for k in range(len(visited_states) - i):
                    self.Gt_states[visited_states[i]] += self.discount_factor ** k * reward_traced[i+k]
    # backward update
    def cal_Gt_and_update_states_value(self,visited_states, reward_traced, first_visited_key):
        trace_len = len(visited_states)
        Gt = 0
        for i in range(trace_len-1,-1,-1):
            Gt = self.discount_factor * Gt + reward_traced[i]
            if i in first_visited_key:
                self.values[visited_states[i]] = self.values[visited_states[i]] + (Gt - self.values[visited_states[i]]) / self.visit_states_count[visited_states[i]]
    
    def run(self) -> None:
        """Run the algorithm until max_episode"""
        # TODO: Update self.values with first-visit Monte-Carlo method
        init_state = self.grid_world.reset()
        while self.episode_counter < self.max_episode:
            reward_traced = []
            visited_states = []
            first_visited_key = []
            key = 0
            self.Gt_states = np.zeros(self.state_space)

            self.visit_states_count[init_state] += 1
            visited_states.append(init_state)
            first_visited_key.append(key)
            while True:
                next_state, reward, done = self.collect_data()
                reward_traced.append(reward)
                
                if next_state not in visited_states:  # first visit
                    self.visit_states_count[next_state] += 1  # count the visit times
                    first_visited_key.append(key+1) # key +1 for the next-state key
                
                if not done:
                    visited_states.append(next_state)
                    key += 1
                else:
                    # forward update
                    # self.cal_Gt_states(visited_states, reward_traced, first_visited_key)
                    # self.update_states_value(visited_states, first_visited_key)
                    # backward update
                    self.cal_Gt_and_update_states_value(visited_states, reward_traced, first_visited_key)
                    break
        
class TDPrediction(ModelFreePrediction):
    def __init__(
            self, grid_world: GridWorld,learning_rate: float, policy: np.ndarray = None, discount_factor: float = 1.0, max_episode: int = 300, seed: int = 1):
        """Constructor for Temporal Difference(0) Prediction

        Args:
            grid_world (GridWorld): GridWorld object
            policy (np.ndarray): Stochastic policy representing action probabilities [state_space, action_space]
            discount_factor (float, optional): Discount factor gamma. Defaults to 1.0.
            max_episdoe (int, optional): Maximum episdoe for data collection. Defaults to 10000.
            learning_rate (float): learning rate for updating state value
        """
        super().__init__(grid_world,policy, discount_factor, max_episode, seed)
        self.lr     = learning_rate
    
    def update_state_value(self, state_c, state_n, reward, flag):
        if flag:
            self.values[state_c] += self.lr*(reward - self.values[state_c]) # self.values[state_n] = 0 ? 
        else:
            self.values[state_c] += self.lr*(reward + self.discount_factor*self.values[state_n] - self.values[state_c])

    def run(self) -> None:
        """Run the algorithm until max episode"""
        # TODO: Update self.values with TD(0) Algorithm
        current_state = self.grid_world.reset()
        while self.episode_counter < self.max_episode:
            while True:
                next_state, reward, done = self.collect_data()
                self.update_state_value(current_state, next_state, reward, done)
                current_state = next_state
                if done:
                    break

class NstepTDPrediction(ModelFreePrediction):
    def __init__(
            self, grid_world: GridWorld, learning_rate: float, num_step: int, policy: np.ndarray = None, discount_factor: float = 1.0, max_episode: int = 300, seed: int = 1):
        """Constructor for Temporal Difference(0) Prediction

        Args:
            grid_world (GridWorld): GridWorld object
            policy (np.ndarray): Stochastic policy representing action probabilities [state_space, action_space]
            discount_factor (float, optional): Discount factor gamma. Defaults to 1.0.
            max_episdoe (int, optional): Maximum episdoe for data collection. Defaults to 10000.
            learning_rate (float): learning rate for updating state value
            num_step (int): n_step look ahead for TD
        """
        super().__init__(grid_world,policy, discount_factor, max_episode, seed)
        self.lr     = learning_rate
        self.n      = num_step

    def cal_Gt_states(self,visited_states, reward_traced,n,flag):
        Gt = 0
        # if n == 0: # terminal state
        #     Gt += reward_traced[0]
        #     return Gt
        
        for i in range(n): # i = 0,1,2
            Gt += self.discount_factor ** i * reward_traced[i]
        
        if not flag:
            Gt += self.discount_factor ** n * self.values[visited_states[n]]
        return Gt
    
    
    def update_state_value(self, states, rewards,n, flag):
        Gt = self.cal_Gt_states(states, rewards,n, flag)
        self.values[states[0]] += self.lr*(Gt-self.values[states[0]])

    def run(self) -> None:
        """Run the algorithm until max_episode"""
        # TODO: Update self.values with N-step TD Algorithm
        current_state = self.grid_world.reset()
        while self.episode_counter < self.max_episode:
            states = [] # first state is the current state
            rewards = []
            steps = 0
            states.append(current_state)
            while True:
                next_state, reward, done = self.collect_data()
                states.append(next_state) # # = 4
                rewards.append(reward) # # = 3
                steps += 1
                if steps >= self.n:
                    if done:   
                        for i in range(self.n): # 0,1,2
                            self.update_state_value(states, rewards, self.n-i, done)
                            states.pop(0)
                            rewards.pop(0)
                        break
                    self.update_state_value(states, rewards, self.n, done)
                    states.pop(0)
                    rewards.pop(0)

# =========================== 2.2 model free control ===========================
class ModelFreeControl:
    """
    Base class for model free control algorithms 
    """

    def __init__(self, grid_world: GridWorld, discount_factor: float = 1.0):
        """
        Args:
            grid_world (GridWorld): GridWorld object
            discount_factor (float, optional): Discount factor gamma. Defaults to 1.0.
        """
        self.grid_world = grid_world
        self.discount_factor = discount_factor
        self.action_space = grid_world.get_action_space()
        self.state_space  = grid_world.get_state_space()
        self.q_values     = np.zeros((self.state_space, self.action_space))  
        self.policy       = np.ones((self.state_space, self.action_space)) / self.action_space # stocastic policy
        self.policy_index = np.zeros(self.state_space, dtype=int)                              # deterministic policy
        self.rng = np.random.default_rng(1)      # only call this in collect_data()
        self.iter_episode = 0

    def get_policy_index(self) -> np.ndarray:
        """Return the policy

        Returns:
            np.ndarray: policy_index
        """
        for s_i in range(self.state_space):
            self.policy_index[s_i] = self.q_values[s_i].argmax()
        return self.policy_index
    
    def get_max_state_values(self) -> np.ndarray:
        max_values = np.zeros(self.state_space)
        for i in range(self.state_space):
            max_values[i] = self.q_values[i].max()
        return max_values
    
    def collect_data(self) -> tuple:
        """
        Use the stochastic policy to interact with the environment and collect one step of data.
        Samples an action based on the action probability distribution for the current state.
        """
        current_state = self.grid_world.get_current_state()  # Get the current state
        # Sample an action based on the stochastic policy's probabilities for the current state
        action_probs = self.policy[current_state]  
        action = self.rng.choice(self.action_space, p=action_probs)  
        # action = np.random.choice(self.action_space, p=action_probs)  
        next_state, reward, done = self.grid_world.step(action)  

        return action, next_state, reward, done

class MonteCarloPolicyIteration(ModelFreeControl):
    def __init__(
            self, grid_world: GridWorld, discount_factor: float, learning_rate: float, epsilon: float):
        """Constructor for MonteCarloPolicyIteration

        Args:
            grid_world (GridWorld): GridWorld object
            discount (float): discount factor gamma
            learning_rate (float): learning rate for updating state value
            epsilon (float): epsilon-greedy threshold
        """
        super().__init__(grid_world, discount_factor)
        self.lr      = learning_rate
        self.epsilon = epsilon
        # for plot learning curve & loss curve
        # self.reward_buffer = deque(maxlen = 10)
        # self.loss_buffer = deque(maxlen = 10)
        # self.reward_total = 0
        # self.loss_total = 0
    
    def policy_evaluation(self, state_trace, action_trace, reward_trace) -> None:
        """Evaluate the policy and update the values after one episode"""
        # TODO: Evaluate state value for each Q(s,a)
        # forward update
        # for i in range(len(state_trace)-1): # last state is initial state
        #     Gt = 0
        #     for j in range(len(state_trace) -1 - i):
        #         Gt += self.discount_factor ** j * reward_trace[i+j]
        #     
        #     # update Q(s,a)
        #     self.q_values[state_trace[i]][action_trace[i]] += self.lr*(Gt - self.q_values[state_trace[i]][action_trace[i]])
        
        # backward update
        trace_len = len(state_trace)
        Gt = 0
        for i in range(trace_len-1-1,-1,-1): # last state is initial state : from terminal state to initial state(0)
            Gt = self.discount_factor * Gt + reward_trace[i]
            # update Q(s,a)
            self.q_values[state_trace[i]][action_trace[i]] += self.lr*(Gt - self.q_values[state_trace[i]][action_trace[i]])
            # for plot
            # self.loss_total += abs(Gt - self.q_values[state_trace[i]][action_trace[i]]) 

    def policy_improvement(self) -> None:
        """Improve policy based on Q(s,a) after one episode"""
        # TODO: Improve the policy
        self.get_policy_index()
        # epsilon greedy
        m = self.action_space
        for i in range(self.state_space):
            for j in range(m):
                if j == self.policy_index[i]:
                    self.policy[i][j] = 1 - self.epsilon + self.epsilon/m
                else:
                    self.policy[i][j] = self.epsilon/m  
            
    def run(self, max_episode=1000) -> None:
        """Run the algorithm until convergence."""
        # TODO: Implement the Monte Carlo policy evaluation with epsilon-greedy
        current_state = self.grid_world.reset()
        
        # for plot
        # wandb.login()
        # run = wandb.init(
        #     project="rl-hw2",
        # )
 
        while self.iter_episode < max_episode:
            # TODO: write your code here
            # hint: self.grid_world.reset() is NOT needed here
            self.iter_episode += 1
            state_trace   = [current_state]
            action_trace  = []
            reward_trace  = []
            # for plot
            # if self.iter_episode > 10:
            #     R = sum(self.reward_buffer)/10
            #     L = sum(self.loss_buffer)/10
            #     # wandb.log({f"Learning curves : epsilon = {self.epsilon}" : R})
            #     # wandb.log({f"Learning curves : epsilon = {self.epsilon}" : R, f"Loss curves : epsilon = {self.epsilon}" : L})
            #     wandb.log({"Learning curves" : R, "Loss curves" : L})
            # self.reward_total = 0
            # self.loss_total = 0

            while True:
                action, next_state, reward, done = self.collect_data()
                action_trace.append(action)
                state_trace.append(next_state)
                reward_trace.append(reward)
                # for plot
                # self.reward_total += reward

                if done:
                    self.policy_evaluation(state_trace, action_trace, reward_trace)
                    self.policy_improvement()
                    current_state = next_state
                    # for plot
                    # reward_avg = self.reward_total/self.grid_world.get_step_count()
                    # loss_avg = self.loss_total/self.grid_world.get_step_count()
                    # self.reward_buffer.append(reward_avg)
                    # self.loss_buffer.append(loss_avg)
                    break 


class SARSA(ModelFreeControl):
    def __init__(
            self, grid_world: GridWorld, discount_factor: float, learning_rate: float, epsilon: float):
        """Constructor for SARSA

        Args:
            grid_world (GridWorld): GridWorld object
            discount (float): discount factor gamma
            learning_rate (float): learning rate for updating state value
            epsilon (float): epsilon-greedy threshold
        """
        super().__init__(grid_world, discount_factor)
        self.lr      = learning_rate
        self.epsilon = epsilon
        # for plot learning curve & loss curve
        # self.reward_buffer = deque(maxlen = 10)
        # self.loss_buffer = deque(maxlen = 10)
        # self.reward_total = 0
        # self.loss_total = 0
        

    def policy_eval_improve(self, s, a, r, s2, a2, is_done) -> None:
        """Evaluate the policy and update the values after one step"""
        # TODO: Evaluate Q value after one step and improve the policy
        if is_done:
            self.q_values[s][a] += self.lr*(r - self.q_values[s][a])
            # for plot
            # self.loss_total += abs(r - self.q_values[s][a])
        else:
            self.q_values[s][a] += self.lr*(r + self.discount_factor*self.q_values[s2][a2] - self.q_values[s][a])
            # for plot
            # self.loss_total += abs(r + self.discount_factor*self.q_values[s2][a2] - self.q_values[s][a])

        # improve policy
        self.get_policy_index()
        # epsilon greedy
        m = self.action_space
        for j in range(m):
            if j == self.policy_index[s]:
                self.policy[s][j] = 1 - self.epsilon + self.epsilon/m
            else:
                self.policy[s][j] = self.epsilon/m  
       
    def run(self, max_episode=1000) -> None:
        """Run the algorithm until convergence."""
        # TODO: Implement the TD policy evaluation with epsilon-greedy
        current_state = self.grid_world.reset()
        # for plot
        # wandb.login()
        # run = wandb.init(
        #     project="rl-hw2",
        # )

        while self.iter_episode < max_episode:
            self.iter_episode += 1
            # TODO: write your code here
            # hint: self.grid_world.reset() is NOT needed here
            # for plot
            # if self.iter_episode > 10:
            #     R = sum(self.reward_buffer)/10
            #     L = sum(self.loss_buffer)/10
            #     # wandb.log({f"Learning curves : epsilon = {self.epsilon}" : R})
            #     # wandb.log({f"Learning curves : epsilon = {self.epsilon}" : R, f"Loss curves : epsilon = {self.epsilon}" : L})
            #     wandb.log({"Learning curves" : R, "Loss curves" : L})

            # self.reward_total = 0
            # self.loss_total = 0
            while True:
                if self.iter_episode == 1:
                    current_action, next_state, reward, done = self.collect_data()
                else:
                    next_state, reward, done = self.grid_world.step(current_action)  
                # for plot
                # self.reward_total += reward

                # next_action = self.policy[next_state].argmax() # deterministic policy
                action_probs = self.policy[next_state]  
                next_action = self.rng.choice(self.action_space, p=action_probs) # stochastic policy  
               
                self.policy_eval_improve(current_state, current_action, reward, next_state, next_action, done)   
                current_state = next_state
                current_action = next_action
                if done:
                    # for plot
                    # reward_avg = self.reward_total/self.grid_world.get_step_count()
                    # loss_avg = self.loss_total/self.grid_world.get_step_count()
                    # self.reward_buffer.append(reward_avg)
                    # self.loss_buffer.append(loss_avg)
                    break

class Q_Learning(ModelFreeControl):
    def __init__(
            self, grid_world: GridWorld, discount_factor: float, learning_rate: float, epsilon: float, buffer_size: int, update_frequency: int, sample_batch_size: int):
        """Constructor for Q_Learning

        Args:
            grid_world (GridWorld): GridWorld object
            discount (float): discount factor gamma
            learning_rate (float): learning rate for updating state value
            epsilon (float): epsilon-greedy threshold
        """
        super().__init__(grid_world, discount_factor)
        self.lr                = learning_rate
        self.epsilon           = epsilon
        self.buffer            = deque(maxlen=buffer_size)
        self.update_frequency  = update_frequency
        self.sample_batch_size = sample_batch_size
        # for plot learning curve & loss curve
        # self.reward_buffer = deque(maxlen = 10)
        # self.loss_buffer = deque(maxlen = 10)
        # self.reward_total = 0
        # self.loss_total = 0

    def add_buffer(self, s, a, r, s2, d) -> None:
        # TODO: add new transition to buffer
        self.buffer.append((s, a, r, s2, d))

    def sample_batch(self) -> list:
        # TODO: sample a batch of index of transitions from the buffer
        batch = []
        buffer_index = np.random.randint(len(self.buffer), size=self.sample_batch_size)
        for i in buffer_index:
            batch.append(self.buffer[i])
        return batch

    def policy_eval_improve(self, s, a, r, s2, a2, is_done) -> None:
        """Evaluate the policy and update the values after one step"""
        #TODO: Evaluate Q value after one step and improve the policy
        if is_done:
            self.q_values[s][a] += self.lr*(r - self.q_values[s][a])
            # for plot
            # self.loss_total += abs(r - self.q_values[s][a])
        else:
            self.q_values[s][a] += self.lr*(r + self.discount_factor*self.q_values[s2][a2] - self.q_values[s][a])
            # for plot
            # self.loss_total += abs(r + self.discount_factor*self.q_values[s2][a2] - self.q_values[s][a])        # improve policy
        self.get_policy_index()
        # epsilon greedy
        m = self.action_space
        for j in range(m):
            if j == self.policy_index[s]:
                self.policy[s][j] = 1 - self.epsilon + self.epsilon/m
            else:
                self.policy[s][j] = self.epsilon/m  

    def run(self, max_episode=1000) -> None:
        """Run the algorithm until convergence."""
        # TODO: Implement the Q_Learning algorithm
        current_state = self.grid_world.reset()
        transition_count = 0
        batch = []

        # for plot
        # wandb.login()
        # run = wandb.init(
        #     project="rl-hw2",
        # )

        while self.iter_episode < max_episode:
            # TODO: write your code here
            self.iter_episode += 1
            # for plot
            # if self.iter_episode > 10:
            #     R = sum(self.reward_buffer)/10
            #     L = sum(self.loss_buffer)/10
            #     # wandb.log({f"Learning curves : epsilon = {self.epsilon}" : R})
            #     # wandb.log({f"Learning curves : epsilon = {self.epsilon}" : R, f"Loss curves : epsilon = {self.epsilon}" : L})
            #     wandb.log({"Learning curves" : R, "Loss curves" : L})
            # self.reward_total = 0
            # self.loss_total = 0

            while True:
                current_action, next_state, reward, done = self.collect_data()
                # add transition to buffer
                self.add_buffer(current_state, current_action, reward, next_state, done)
                transition_count += 1
                
                # for plot
                # self.reward_total += reward

                if transition_count % self.update_frequency == 0:
                    # sample a batch from buffer
                    batch = self.sample_batch()

                    for i in range(len(batch)):
                        (s, a, r, s2, d) = batch[i]
                        a2 = self.q_values[s2].argmax()
                        self.policy_eval_improve(s, a, r, s2, a2, d) # no matter done or not, finish the update from buffer
                
                current_state = next_state
                if done:
                    # for plot
                    # reward_avg = self.reward_total/self.grid_world.get_step_count()
                    # loss_avg = self.loss_total/self.grid_world.get_step_count()
                    # self.reward_buffer.append(reward_avg)
                    # self.loss_buffer.append(loss_avg)
                    break