from pendulum import *
from ActorCritic import ActorCritic
import matplotlib.pyplot as plt
from train import train

def main():
    
    env = PendulumEnv()
    num_episodes = 5
    num_of_time_steps = 200
    
    #1    
    policy1 = ActorCritic(env, alpha_value=0.0001, alpha_policy=0.01, gamma=0.99, sigma = 1.5)
    reward_plot1 = train(env, policy1, num_episodes, num_of_time_steps)
    plt.figure()
    plt.plot(reward_plot1)
    plt.xlabel("Every 10th epsiode")
    plt.ylabel("Sum of rewards in the episode")
    plt.title("1 > Policy Step Size > Value Step Size")    
    #plt.savefig('1-3_1.png')
    plt.show()

    #2
    policy2 = ActorCritic(env, alpha_value=0.1, alpha_policy=0.01, gamma=0.99, sigma = 1.5)
    reward_plot2 = train(env, policy2, num_episodes, num_of_time_steps)
    plt.figure()
    plt.plot(reward_plot2)
    plt.xlabel("Every 10th epsiode")
    plt.ylabel("Sum of rewards in the episode")
    plt.title("1 > Policy Step Size < Value Step Size")    
    #plt.savefig('1-3_2.png')
    plt.show()

    #3
    policy3 = ActorCritic(env, alpha_value=0.001, alpha_policy=0.001, gamma=0.99, sigma = 1.5)
    reward_plot3 = train(env, policy3, num_episodes, num_of_time_steps)
    plt.figure()
    plt.plot(reward_plot3)
    plt.xlabel("Every 10th epsiode")
    plt.ylabel("Sum of rewards in the episode")
    plt.title("1 > Policy Step Size = Value Step Size")    
    #plt.savefig('1-3_3.png')
    plt.show()

    #4
    policy4 = ActorCritic(env, alpha_value=1, alpha_policy=0.1, gamma=0.99, sigma = 1.5)
    reward_plot4 = train(env, policy4, num_episodes, num_of_time_steps)
    plt.figure()
    plt.plot(reward_plot4)
    plt.xlabel("Every 10th epsiode")
    plt.ylabel("Sum of rewards in the episode")
    plt.title("Policy Step Size > Value Step Size > 1")    
    plt.show()
    #plt.savefig('1-3_4.png')
    #5
    plt.figure()
    plt.plot(reward_plot1, label = "1 > alpha_Policy > alpha_Value")
    plt.plot(reward_plot2, label = "1 > alpha_Policy < alpha_Value")
    plt.plot(reward_plot3, label = "1 > alpha_Policy = alpha_Value")
    plt.plot(reward_plot4, label = "alpha_Policy > alpha_Value > 1")
    plt.title("Performance for different values of policy and value step size")
    plt.xlabel("Every 10th epsiode")
    plt.ylabel("Sum of rewards in the episode")
    plt.legend()
    #plt.savefig("all_plots.png")
    plt.show()

if __name__ == "__main__":
    main()




