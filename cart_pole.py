import gym
import random
import tensorflow as tf
import numpy as np

env = gym.make('CartPole-v0')

NUM_ITERATIONS = 10000

ALPHA = .1
GAMMA = .9
NUM_FLOAT_FEATURES = 4
LEARNING_RATE = 0.01

class TrainerState(object):
  def __init__(self):
    self.input_features = tf.placeholder(tf.float32, shape=[1, NUM_FLOAT_FEATURES])
    self.output_labels = tf.placeholder(tf.float32, shape=[1, 2])
    # Matrix [[update_mag_0, 0], [0, update_mag_1]]
    self.update_matrix = tf.placeholder(tf.float32, shape=[2,2])
    self.W = tf.Variable(tf.zeros([NUM_FLOAT_FEATURES, 2]))
    self.y = tf.matmul(self.input_features, self.W)
    add_delta = tf.transpose(tf.matmul(
        self.update_matrix, tf.concat([self.input_features, self.input_features], 0)))
    self.update_weights = self.W.assign_add(add_delta)
    self.sess = tf.Session()
    initializer = tf.global_variables_initializer()
    self.sess.run(initializer)


def UpdateModel(ts, observations, actions, updates):
  for i in range(len(observations)):
    update_matrix = np.zeros(shape=[2,2])
    update_matrix[actions[i], actions[i]] = updates[i] * LEARNING_RATE
    ts.sess.run(ts.update_weights, feed_dict={
        ts.input_features:[observations[i]],
        ts.update_matrix:update_matrix})

def RunTrainer():
  trainer_state = TrainerState()
  for i in range(NUM_ITERATIONS):
    obs, act, upd = RunIteration(trainer_state, i)
    UpdateModel(trainer_state, obs, act, upd)
  obs, act, upd = RunIteration(trainer_state, i, render=True)


def DoExplore():
  return random.random() < ALPHA

def RunIteration(ts, iteration, render=False):
  observation = env.reset()
  observations = []
  observations.append(observation)
  actions = []
  rewards = []
  updates = []
  done = False
  choices = ts.sess.run(ts.y, feed_dict={ts.input_features:[observation]})[0]
  while not done:
    if DoExplore():
      action = env.action_space.sample()
    else:
      action = np.argmax(choices)
    if render:
      env.render()
    observation, reward, done, info = env.step(action) # take a random action
    actions.append(action)
    if len(observations) > 0:
      old_q = choices[actions[-1]]
      choices = ts.sess.run(ts.y, feed_dict={ts.input_features:[observation]})[0]
      if not done:
        updates.append(reward + GAMMA * np.max(choices) - old_q)
        if render:
          print "======="
          print "old_q:%f" % old_q
          print "max:%f" % np.max(choices)
          print "update:%f" % updates[-1]
      else:
        # Terminal state reward is 0.
        updates.append(-old_q)
    rewards.append(reward)
    if done:
      break
    observations.append(observation)
  return observations, actions, updates

if __name__ == "__main__":
  RunTrainer()
