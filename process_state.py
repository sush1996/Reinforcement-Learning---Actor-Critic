from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import FeatureUnion
from sklearn.kernel_approximation import RBFSampler
import numpy as np
import copy
import gym

env_sampler = gym.make('Pendulum-v0')
state_space_samples = np.array([env_sampler.observation_space.sample() for x in range(20000)])
scaler = StandardScaler()
scaler.fit(state_space_samples)

# Used to convert a state to a featurizes represenation.
# RBF kernels with different variances to cover different parts of the space
featurizer = FeatureUnion([
    ("rbf1", RBFSampler(gamma=5.0, n_components=100)),
    ("rbf2", RBFSampler(gamma=2.0, n_components=100))
])
featurizer.fit(scaler.transform(state_space_samples))

def process_state(state):
    
    scaled = scaler.transform([state])
    featurized = featurizer.transform(scaled)
    
    return featurized[0]
