# reinforce_dqn_taxi
Deep Q-Learning (DQN) for OpenAI Taxi Domain.

Implementation utilizes a target network to guide learning in the right direction and takes advantage of experience replay to remove state transition dependencies. The Markov Decision Process and overall environment are defined/provided by OpenAI.

Notes:

Empircally, running the DQN model with multiple passes (saving weights from previous pass and running model again initialized with those weights) leads to better performance because the exploration/exploitation epsilon constant is allowed to re-decay, effectively helping the agent escape from local "traps" and not get stuck during training. 

Tensorboard was also integrated into this project for training/progress visualizations.
