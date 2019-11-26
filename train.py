
# Training the actor-critic and subsequently getting the reward acumulated over episodes
def train(env, model, num_episodes, num_of_time_steps):
    
    reward_plot = []
    
    for i in range(num_episodes):
        model.reset_episode()
        done = False
        j = 0
        state = env.reset()
        total_reward = 0
        
        while (not done) and (j < num_of_time_steps):            
            action = model.act(state)
            next_state, reward, done, _ = env.step(action) 
            model.update(state, action, reward, next_state, done)
            total_reward += reward
            state = next_state
        
        print("Episode:", i, "Total Reward:", total_reward)
        
        if i%10==0:
            reward_plot = reward_plot + [total_reward]
        
    return reward_plot
    
