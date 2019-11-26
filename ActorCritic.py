import numpy as np
from process_state import process_state

#Actor Critic Class - Actor: Policy and Critic: Value

class ActorCritic(object):

    # You may add extra parameters to this function to help with discretization
    # Also, you will need to tune the sigma value and learning rates
    def __init__(self, env, gamma, sigma, alpha_value, alpha_policy):
        
        # Upper and lower limits of the state 
        self.min_state = env.min_state
        self.max_state = env.max_state
        self.num_states = 200
        
        self.value_weights = np.zeros((self.num_states, 1))
        self.policy_weights = np.zeros((self.num_states,1))
    
        self.gamma = gamma
        self.sigma = sigma

        # Step sizes for the value function and policy
        self.alpha_value = alpha_value
        self.alpha_policy = alpha_policy
 
    # This function should return an action given the
    # state by evaluating the Gaussian polic
    def act(self, state):
        processed_state = process_state(state)
        processed_state = processed_state.reshape((self.num_states, 1))
        mu = np.dot(self.policy_weights.T, processed_state)
        
        mu = mu[0]
        action = np.random.normal(mu, self.sigma) 
        
        return action

    #takes in a state and predicts a value for the state
    def predict_value(self,state):
        processed_state = process_state(state)
        processed_state = processed_state.reshape((self.num_states, 1))
        value_estimate = np.sum(np.dot(self.value_weights.T, processed_state))
        return value_estimate

    def update_value_function(self, state, advantage, value_step_size):
        processed_state = process_state(state)
        processed_state = processed_state.reshape((self.num_states, 1))
        value_grad = processed_state
        self.value_weights = self.value_weights + (value_step_size * value_grad * advantage)

    def update_policy(self, state, action, advantage, step_size):
        processed_state = process_state(state)
        processed_state = processed_state.reshape((self.num_states, 1))
        mu = np.dot(self.policy_weights.T,processed_state)[0][0]
        scalar_diff = action[0] - mu
        scalar_mult = scalar_diff/float((self.sigma**2))
        grad = processed_state * scalar_mult
        
        self.policy_weights = self.policy_weights +  step_size * grad * advantage * self.I
    
    #1) Computes the value function gradient
    #2) Computes the policy gradient
    #3) Performs the gradient step for the value and policy functions
    
    def update(self, state, action, reward, next_state, done):
        advantage = 0 
        state_val = self.predict_value(state)
        
        if(done):
            advantage = reward - state_val
        else: 
            advantage = reward + (self.gamma * self.predict_value(next_state)) - self.predict_value(state)
        
        self.update_value_function(state, advantage, self.alpha_value)
        self.update_policy(state, action, advantage, self.alpha_policy)
        self.I = self.I * self.gamma
    
    def reset_episode(self):
        self.I = 1
