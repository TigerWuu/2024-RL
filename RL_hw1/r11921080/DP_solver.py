import numpy as np
import queue

from gridworld import GridWorld


class DynamicProgramming:
    """Base class for dynamic programming algorithms"""

    def __init__(self, grid_world: GridWorld, discount_factor: float = 1.0):
        """Constructor for DynamicProgramming

        Args:
            grid_world (GridWorld): GridWorld object
            discount_factor (float, optional): Discount factor gamma. Defaults to 1.0.
        """
        self.grid_world = grid_world
        self.discount_factor = discount_factor
        self.threshold = 1e-4  # default threshold for convergence
        self.values = np.zeros(grid_world.get_state_space())  # V(s)
        self.policy = np.zeros(grid_world.get_state_space(), dtype=int)  # pi(s)

    def set_threshold(self, threshold: float) -> None:
        """Set the threshold for convergence

        Args:
            threshold (float): threshold for convergence
        """
        self.threshold = threshold

    def get_policy(self) -> np.ndarray:
        """Return the policy

        Returns:
            np.ndarray: policy
        """
        return self.policy

    def get_values(self) -> np.ndarray:
        """Return the values

        Returns:
            np.ndarray: values
        """
        return self.values

    def get_q_value(self, state: int, action: int) -> float:
        """Get the q-value for a state and action

        Args:
            state (int)
            action (int)

        Returns:
            float
        """
        # TODO: Get reward from the environment and calculate the q-value
        Vk = self.get_values()
        P = 1 # deterministic
        next_state, reward, done = self.grid_world.step(state, action) # action is not 0,1,2,3?
        Vq = reward + self.discount_factor *P* Vk[next_state]*(1-done)
        return Vq, next_state, done


class IterativePolicyEvaluation(DynamicProgramming):
    def __init__(
        self, grid_world: GridWorld, policy: np.ndarray, discount_factor: float
    ):
        """Constructor for IterativePolicyEvaluation

        Args:
            grid_world (GridWorld): GridWorld object
            policy (np.ndarray): policy (probability distribution state_spacex4)
            discount (float): discount factor gamma
        """
        super().__init__(grid_world, discount_factor)
        self.policy = policy

    def get_state_value(self, state: int) -> float:
        """Get the value for a state

        Args:
            state (int)

        Returns:
            float: value
        """
        # TODO: Get the value for a state by calculating the q-values
        value = self.get_values()[state] 
        return value

    def evaluate(self):
        """Evaluate the policy and update the values for one iteration"""
        # TODO: Implement the policy evaluation step
        new_values = np.zeros(self.get_values().shape)
        
        for s in range(self.grid_world.get_state_space()):
            for a in range(self.grid_world.get_action_space()):
                vq, _, _ = self.get_q_value(s, a)
                new_values[s] += self.policy[s][a] * vq
        
        new_delta = np.linalg.norm(new_values - self.get_values())
        # print(new_delta)
        self.values = new_values
        return new_delta

    def run(self) -> None:
        """Run the algorithm until convergence."""
        # TODO: Implement the iterative policy evaluation algorithm until convergence
        while True:
            new_delta = self.evaluate()
            if new_delta < self.threshold:
                break
        # raise NotImplementedError


class PolicyIteration(DynamicProgramming):
    def __init__(self, grid_world: GridWorld, discount_factor: float = 1.0):
        """Constructor for PolicyIteration

        Args:
            grid_world (GridWorld): GridWorld object
            discount_factor (float, optional): Discount factor gamma. Defaults to 1.0.
        """
        super().__init__(grid_world, discount_factor)
        # self.policy = np.random.randint(0, 4, grid_world.get_state_space())

    def get_state_value(self, state: int) -> float: 
        """Get the value for a state

        Args:
            state (int)

        Returns:
            float
        """
        # TODO: Get the value for a state by calculating the q-values
        value = self.get_values()[state] 
        return value
    
    def policy_evaluation(self):
        """Evaluate the policy and update the values"""
        # TODO: Implement the policy evaluation step
        new_values = np.zeros(self.get_values().shape)
        
        # for s in range(self.grid_world.get_state_space()):
        #     for a in range(self.grid_world.get_action_space()):
        #         new_values[s] += self.policy[s][a] * self.get_q_value(s, a)
  
        for s in range(self.grid_world.get_state_space()):
            
            new_values[s], _, _ = self.get_q_value(s, self.policy[s])
       
        delta = np.linalg.norm(new_values - self.get_values())
        self.values = new_values
        return delta


    def policy_improvement(self):
        """Improve the policy based on the evaluated values"""
        # TODO: Implement the policy improvement step
        for s in range(self.grid_world.get_state_space()):
            best_action = 0
            best_value = -np.inf
            for a in range(self.grid_world.get_action_space()):
                val, _, _ = self.get_q_value(s, a)
                if val > best_value:
                    best_action = a
                    best_value = val

            self.policy[s] = best_action
        
        # for s in range(self.grid_world.get_state_space()):
        #     best_action = np.argmax([self.get_q_value(s, a) for a in range(self.grid_world.get_action_space())])
        #     self.policy[s] = best_action

    def run(self) -> None:
        """Run the algorithm until convergence"""
        # TODO: Implement the policy iteration algorithm until convergence
        while True:
            delta = self.policy_evaluation()
            self.policy_improvement()
            # print(new_delta)
            if delta < self.threshold:
                break


class ValueIteration(DynamicProgramming):
    def __init__(self, grid_world: GridWorld, discount_factor: float = 1.0):
        """Constructor for ValueIteration

        Args:
            grid_world (GridWorld): GridWorld object
            discount_factor (float, optional): Discount factor gamma. Defaults to 1.0.
        """
        super().__init__(grid_world, discount_factor)

    def get_state_value(self, state: int) -> float:
        """Get the value for a state

        Args:
            state (int)

        Returns:
            float
        """
        # TODO: Get the value for a state by calculating the q-values
        value = self.get_values()[state] 
        return value

    def policy_evaluation(self):
        """Evaluate the policy and update the values"""
        # TODO: Implement the policy evaluation step
        new_values = np.zeros(self.get_values().shape)
        
        # for s in range(self.grid_world.get_state_space()):
        #     best_action = np.argmax([self.get_q_value(s, a) for a in range(self.grid_world.get_action_space())])
        #     new_values[s] = self.get_q_value(s, best_action)
        #     self.policy[s] = best_action
        
        for s in range(self.grid_world.get_state_space()):
            best_action = 0
            best_value = -np.inf
            for a in range(self.grid_world.get_action_space()):
                val, _, _ = self.get_q_value(s, a)
                if val > best_value:
                    best_action = a
                    best_value = val

            new_values[s] = best_value
            self.policy[s] = best_action
            
        new_delta = np.linalg.norm(new_values - self.get_values())
        self.values = new_values
        return new_delta

    def run(self) -> None:
        """Run the algorithm until convergence"""
        # TODO: Implement the value iteration algorithm until convergence
        while True:
            new_delta = self.policy_evaluation()
            # print(new_delta)
            if new_delta < self.threshold:
                break
        
        # for s in range(self.grid_world.get_state_space()):
        #     best_action = np.argmax([self.get_q_value(s, a) for a in range(self.grid_world.get_action_space())])
        #     self.policy[s] = best_action

class AsyncDynamicProgramming(DynamicProgramming):
    def __init__(self, grid_world: GridWorld, discount_factor: float = 1.0):
        """Constructor for ValueIteration

        Args:
            grid_world (GridWorld): GridWorld object
            discount_factor (float, optional): Discount factor gamma. Defaults to 1.0.
        """
        super().__init__(grid_world, discount_factor)

    def in_place_dp(self):
        count = 0
        # for s in range(self.grid_world.get_state_space()):
        #     best_action = np.argmax([self.get_q_value(s, a) for a in range(self.grid_world.get_action_space())])
        #     self.values[s] = self.get_q_value(s, best_action)
        #     self.policy[s] = best_action
        while True:
            new_values = self.get_values().copy() # for calculating value delta 
            for s in range(self.grid_world.get_state_space()):
                best_action = 0
                best_value = -np.inf
                for a in range(self.grid_world.get_action_space()):
                    val, _, _ = self.get_q_value(s, a)
                    if val > best_value:
                        best_action = a
                        best_value = val

                self.values[s] = best_value
                self.policy[s] = best_action

                count += 1 # how many times we update the value
                
            delta = np.linalg.norm(new_values - self.get_values())
            if delta < self.threshold:
                return

    def prioritized_sweeping(self):
        pq = queue.PriorityQueue()
        count = 0
        while True:
            old_values = self.get_values().copy()
            predeccessors = {} # how to update the predeccessors
            for s in range(self.grid_world.get_state_space()):
                best_action = 0
                best_value = -np.inf
                next_state = 0
                for a in range(self.grid_world.get_action_space()):
                    val, state_n, _ = self.get_q_value(s, a)
                    if val > best_value:
                        best_action = a
                        best_value = val
                        next_state = state_n

                if next_state in predeccessors:
                    predeccessors[next_state].append([s, best_action])
                else:
                    predeccessors[next_state] = [[s, best_action]]
                # print(predeccessors)
                delta = abs(best_value - old_values[s]) # if delta = 0, old_values = self.values
                # print("delta:",delta)
                if delta > self.threshold:
                    pq.put((-delta, (s, best_action, best_value)))
            # print(predeccessors)
            while not pq.empty():
                _, (S, A, R) = pq.get()
                # update the value & policy
                # best_value, _, _ = self.get_q_value(S, A)
                self.values[S] = R
                self.policy[S] = A
                if S in predeccessors:
                    for i in range(len(predeccessors[S])):
                        S_ = predeccessors[S][i][0]
                        A_ = predeccessors[S][i][1]
                        val, _, done = self.get_q_value(S_, A_)
                        # val = -1 + 0.9* self.get_values()[S] # reward + discount * next_best_state_value
                        delta_p = abs(val - self.get_values()[S_])
                        # print(delta_p)
                        if delta_p > self.threshold:
                            pq.put((-delta_p, (S_, A_, val)))
                # how to determine the convergence ? 
            value_delta = np.linalg.norm(old_values - self.get_values())
            # print("value:", value_delta)
            if value_delta < self.threshold:
                return
            count += 1
            # print(count)

            #while True:
            #    old_values = self.get_values().copy()
            #    for s in range(self.grid_world.get_state_space()):
            #        best_action = 0
            #        best_value = -np.inf
            #        next_state = 0
            #        predeccessors = {} # how to update the predeccessors
            #        for a in range(self.grid_world.get_action_space()):
            #            val, state_n, _ = self.get_q_value(s, a)
            #            if val > best_value:
            #                best_action = a
            #                best_value = val
            #                next_state = state_n           
            #        if next_state in predeccessors:
            #            predeccessors[next_state].append([s, best_action])
            #        else:
            #            predeccessors[next_state] = [[s, best_action]]
            #        # print(predeccessors)         
            #        delta = abs(best_value - old_values[s]) # if delta = 0, old_values = self.values
            #        print("delta:",delta)
            #        if delta > self.threshold:
            #            pq.put((-delta, (s, best_action)))
            #            while not pq.empty():
            #                _, (S, A) = pq.get()
            #                # update the value & policy
            #                best_value, _, _ = self.get_q_value(S, A)
            #                self.values[S] = best_value
            #                self.policy[S] = A
            #                if S in predeccessors:
            #                    for i in range(len(predeccessors[S])):
            #                        S_ = predeccessors[S][i][0]
            #                        A_ = predeccessors[S][i][1]
            #                        val, _, _ = self.get_q_value(S_, A_)
            #                        delta_p = abs(val - self.get_values()[S_])
            #                        # print(delta_p)
            #                        if delta_p > self.threshold:
            #                            pq.put((-delta_p, (S_, A_)))
            #        # how to determine the convergence ? 
            #        value_delta = np.linalg.norm(old_values - self.get_values())
            #        print("value:", value_delta)
            #        if value_delta < self.threshold:
            #            return
            # count += 1
            # print(count)
            # if count > 10:
            #     return

    def real_time_dp(self):
        s = 0
        count = 0
        old_values = self.get_values().copy() # for calculating value delta 
        while True:
            best_action = 0
            best_value = -np.inf
            next_state = 0
            done = 0
            for a in range(self.grid_world.get_action_space()):
                val, next_s, do = self.get_q_value(s, a)
                if val > best_value:
                    best_action = a
                    best_value = val
                    next_state = next_s
                    done = do   
            
            self.values[s] = best_value
            self.policy[s] = best_action
            s = next_state

            if done == True: # if this is the terminal state, check the convergence
                delta = np.linalg.norm(old_values - self.get_values())
                if delta < self.threshold:
                    return
              
                old_values = self.get_values().copy() # for calculating value delta 
                s = 0
            
            # count += 1
            # print(count)
            # if count > 200:
            #     return

    def run(self) -> None:
        """Run the algorithm until convergence"""
        # TODO: Implement the async dynamic programming algorithm until convergence
        self.in_place_dp()
        # self.prioritized_sweeping()
        # self.real_time_dp()
        # for s in range(self.grid_world.get_state_space()):
        #     best_action = np.argmax([self.get_q_value(s, a) for a in range(self.grid_world.get_action_space())])
        #     self.policy[s] = best_action
