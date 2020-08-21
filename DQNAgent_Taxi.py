#To run this, you need Tensorflow version==1.15. Type "python DQNAgent_Taxi.py" in the terminal to train the agent. The model will automatically save its weights when a certain threshold of performance is reached.
#To render/display the agent interacting with the environment, set SHOW_PREVIEW to True and set AGGREGATE_STATS_EVERY to 1 to see rendering for every timestep. 
#To see training visualizations on Tensorboard, you should type something like "tensorboard --logdir=taxi-1597569936/" in the terminal. taxi-1597569936 should be replaced by the name of the folder in the log folder.
#Then, just copy the link that the terminal gives you into your browser. If this link doesn't work, try searching http://localhost:6006/ 

import numpy as np
import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense, Dropout, Conv2D, MaxPooling2D, Activation, Flatten, Embedding, Reshape
from keras.optimizers import Adam
from keras.callbacks import TensorBoard
from collections import deque
import time
import random
from tqdm import tqdm
import os
from PIL import Image
import cv2
import gym 

DISCOUNT = 0.99
REPLAY_MEMORY_SIZE = 50_000  # How many last steps to keep for model training
MIN_REPLAY_MEMORY_SIZE = 1_000  # Minimum number of steps in a memory to start training
MINIBATCH_SIZE = 64  # number of steps (samples) to use for training
UPDATE_TARGET_EVERY = 5  # Terminal states (end of episodes)
MODEL_NAME = 'taxi'
MIN_REWARD = -300  # For model save: change this to be higher for the later passes
AVG_REWARD = -200 # For model save: change this to be higher for the later passes
MEMORY_FRACTION = 0.20

# Environment settings
EPISODES = 20_000

# Exploration settings
epsilon = 1  # exploration/exploitation term: will be decayed
EPSILON_DECAY = 0.99975	# decay rate
MIN_EPSILON = 0.001	#smallest that epsilon can be

#  Stats settings
AGGREGATE_STATS_EVERY = 50  
SHOW_PREVIEW = False	# To render the environment and visually see the agent train/interact, set this to True
ACTION_SPACE_SIZE = 6

env = gym.make('Taxi-v3')	#import taxi domain environment from OpenAI Gym

# For stats
ep_rewards = [-200]

# For more repetitive results
random.seed(1)
np.random.seed(1)
tf.set_random_seed(1)

# Create models folder
if not os.path.isdir('models'):
	os.makedirs('models')
	
# Create logs folder
if not os.path.isdir('logs'):
	os.makedirs('logs')

# Own Tensorboard class
class ModifiedTensorBoard(TensorBoard):
	# Overriding init to set initial step and writer (one log file for all .fit() calls)
	def __init__(self, **kwargs):
		super().__init__(**kwargs)
		self.step = 1
		self.writer = tf.summary.FileWriter(self.log_dir)

	# Overriding to stop creating default log writer
	def set_model(self, model):
		pass

	# Overrided, saves logs with our step number
	# (otherwise every .fit() will start writing from 0th step)
	def on_epoch_end(self, epoch, logs=None):
		self.update_stats(**logs)

	# We train for one batch only, no need to save anything at epoch end
	def on_batch_end(self, batch, logs=None):
		pass

	# Overrided, do not close writer
	def on_train_end(self, _):
		pass

	# Custom method for saving own metrics
	# Creates writer, writes custom metrics and closes writer
	def update_stats(self, **stats):
		self._write_logs(stats, self.step)

class DQNAgent:
	def __init__(self):

		# Main model
		self.model = self.create_model()

		# Target network
		self.target_model = self.create_model()
		self.target_model.set_weights(self.model.get_weights())

		# An array with last 50_000 steps for training to sample our training batch from
		self.replay_memory = deque(maxlen=REPLAY_MEMORY_SIZE)

		# Custom tensorboard object
		self.tensorboard = ModifiedTensorBoard(log_dir="logs/{}-{}".format(MODEL_NAME, int(time.time())))

		# Used to count when to update target network with main network's weights
		self.target_update_counter = 0

	def create_model(self):
		model = Sequential()
		model.add(Embedding(500, 10, input_length=1))	#env.step(action) for the taxi domain returns an single integer for the resultant state that represents one of 500 possible states. Input length is 1.
		model.add(Reshape((10,)))
		model.add(Dense(50, activation='relu'))
		model.add(Dense(50, activation='relu'))
		model.add(Dense(50, activation='relu'))
		model.add(Dense(ACTION_SPACE_SIZE, activation='linear'))
		model.compile(loss="mse", optimizer=Adam(lr=0.001), metrics=['accuracy'])
		return model

	def update_replay_memory(self, transition):
		self.replay_memory.append(transition)

	def get_qs(self, state):
		return self.model.predict(np.array([state]))

	def train(self, terminal_state, step):
		if len(self.replay_memory) < MIN_REPLAY_MEMORY_SIZE:
			return
		minibatch = random.sample(self.replay_memory, MINIBATCH_SIZE)

		current_states = np.array([transition[0] for transition in minibatch])/500	# Normalize
		current_qs_list = self.model.predict(current_states)

		new_current_states = np.array([transition[3] for transition in minibatch])/500	# Normalize
		future_qs_list = self.target_model.predict(new_current_states)

		X = []	#feature sets (iamges from the game)
		y = []	#label sets (actions we can possibly take)

		for index, (current_state, action, reward, new_current_state, done) in enumerate(minibatch):
			if not done:
				max_future_q = np.max(future_qs_list[index])	#highest Q(s', a') value where s' is the next state
				new_q = reward + DISCOUNT * max_future_q	
			else:
				new_q = reward

				#update Q value for given state
			current_qs = current_qs_list[index]    #current Q(s, a) values for current state
			current_qs[action] = new_q	#update how good the given action at the given current state is
			
			X.append(current_state)
			y.append(current_qs)

		self.model.fit(np.array(X)/500, np.array(y), batch_size = MINIBATCH_SIZE, verbose=0, 
					   shuffle=False, callbacks=[self.tensorboard] if terminal_state else None)

		#updating to determine if we want to update target_model yet
		if terminal_state:
			self.target_update_counter += 1

		if self.target_update_counter > UPDATE_TARGET_EVERY:
			self.target_model.set_weights(self.model.get_weights())
			self.target_update_counter = 0

agent = DQNAgent()

for episode in tqdm(range(1, EPISODES+1), ascii=True, unit="episode"):
	agent.tensorboard.step = episode

	episode_reward = 0
	step = 1
	current_state = env.reset()

	done = False

	while not done:
		#exploitation vs exploration (explore more at first and eventually ease into exploiting)
		if np.random.random() > epsilon:
			action = np.argmax(agent.get_qs(current_state))
		else:
			action = np.random.randint(0, ACTION_SPACE_SIZE)

		new_state, reward, done, info = env.step(action)

		episode_reward += reward

		if SHOW_PREVIEW and not episode % AGGREGATE_STATS_EVERY:
			env.render()

		agent.update_replay_memory((current_state, action, reward, new_state, done))
		agent.train(done, step)

		current_state = new_state
		step += 1

	# Append episode reward to a list and log stats (every given number of episodes)
	ep_rewards.append(episode_reward)
	if not episode % AGGREGATE_STATS_EVERY or episode == 1:
		average_reward = sum(ep_rewards[-AGGREGATE_STATS_EVERY:])/len(ep_rewards[-AGGREGATE_STATS_EVERY:])
		min_reward = min(ep_rewards[-AGGREGATE_STATS_EVERY:])
		max_reward = max(ep_rewards[-AGGREGATE_STATS_EVERY:])
		agent.tensorboard.update_stats(reward_avg=average_reward, reward_min=min_reward, reward_max=max_reward, epsilon=epsilon)

		# Save model, but only when average reward is greater or equal a set value or min reward is greater or equal a set value
		if average_reward >= AVG_REWARD or min_reward >= MIN_REWARD:
			agent.model.save(f'models/{MODEL_NAME}__{max_reward:_>7.2f}max_{average_reward:_>7.2f}avg_{min_reward:_>7.2f}min__{int(time.time())}.model')

	# Decay epsilon
	if epsilon > MIN_EPSILON:
		epsilon *= EPSILON_DECAY
		epsilon = max(MIN_EPSILON, epsilon)
