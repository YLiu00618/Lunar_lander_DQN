"""
An agent to land the “Lunar Lander” in OpenAI gym using deep Q-Network (DQN) with epsilon-greedy exploration.

"""


# import argparse
from collections import deque
import gym
import numpy as np
import random
import tensorflow as tf
import matplotlib.pyplot as plt


DEFAULT_HPARAMS = {
    # Network architecture.
    'num_layers': 3,
    'layer_dim': 512,
    'dropout': 1.0,
    'weight_decay': 1e-4,
    'num_states': 8,
    'num_actions': 4,

    # Training.
    'learning_rate': 1e-5,
    'max_steps_per_episode': 10000,
    'max_num_episodes': 2001,
    'update_slow_target_every_steps': 100,
    'replay_buffer_size': 1e6,
    'batch_size': 1024,
    'save_checkpoint_secs': 10,
    'save_summaries_steps': 10,
    'logdir': 'model',

    # RL.
    'discount': 0.99,

    # Epsilon greedy.
    'epsilon_start': 1.0,
    'epsilon_linear_end': 0.95,
    'num_epsilon_linear_steps': 1e4,
    'epsilon_exp_decay': 0.95,
}

def predict_actions(net, is_training, trainable, reuse, hparams):
  # net: float32 feature of shape [batch_size, k].
  with tf.contrib.framework.arg_scope(
      [tf.contrib.layers.fully_connected],
      trainable=trainable, reuse=reuse,
       weights_regularizer=tf.contrib.layers.l2_regularizer(hparams.weight_decay),
       biases_regularizer=tf.contrib.layers.l2_regularizer(hparams.weight_decay)):
    for i in range(hparams.num_layers):
      net = tf.contrib.layers.fully_connected(net, hparams.layer_dim,
          activation_fn=tf.nn.relu,
          scope='fc_%d' % i)
      net = tf.contrib.layers.dropout(net, keep_prob=hparams.dropout,
          is_training=is_training & trainable)
    actions = tf.contrib.layers.fully_connected(net, hparams.num_actions,
        activation_fn=tf.nn.relu,
        scope='actions')
    return actions


def create_train_op(state, next_state, action, reward, is_not_final,
    is_training, hparams):
  with tf.variable_scope('main_network') as scope:
    q_values = predict_actions(state, is_training=is_training,
        trainable=True, reuse=False,
        hparams=hparams)
    next_q_values = tf.stop_gradient(predict_actions(next_state,
      is_training=is_training, trainable=False, reuse=True, hparams=hparams))

  with tf.variable_scope('slow_network', reuse=False):
      slow_next_q_values = tf.stop_gradient(
          predict_actions(next_state, is_training=is_training,
            trainable=False, reuse=False,
            hparams=hparams))

  main_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES,
      scope='main_network')
  slow_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='slow_network')

  update_slow_vars_ops = []
  for i in range(len(slow_vars)):
    update_slow_vars_ops.append(slow_vars[i].assign(main_vars[i]))
  update_slow_vars_ops = tf.group(*update_slow_vars_ops,
      name='update_slow_vars')

  best_next_q_values = tf.gather_nd(slow_next_q_values,
      tf.stack((tf.range(hparams.batch_size), tf.cast(tf.argmax(next_q_values,
        axis=1), tf.int32)), axis=1))

  target_q_values = (reward + tf.to_float(is_not_final) *
      hparams.discount * best_next_q_values)

  estimated_q_values = tf.gather_nd(q_values,
      tf.stack((tf.range(hparams.batch_size), action), axis=1))

  tf.losses.mean_squared_error(target_q_values, estimated_q_values)

  loss = tf.losses.get_total_loss()

  train_op = tf.train.AdamOptimizer(hparams.learning_rate).minimize(loss)
  return train_op, update_slow_vars_ops, q_values


hparams = tf.contrib.training.HParams(**DEFAULT_HPARAMS)

# placeholders
state_ph = tf.placeholder(dtype=tf.float32, shape=[None, hparams.num_states])
next_state_ph = tf.placeholder(dtype=tf.float32, shape=[None, hparams.num_states])
action_ph = tf.placeholder(dtype=tf.int32, shape=[None])
reward_ph = tf.placeholder(dtype=tf.float32, shape=[None])
is_not_final_ph = tf.placeholder(dtype=tf.float32, shape=[None])
is_training_ph = tf.placeholder(dtype=tf.bool, shape=())

train_op, update_slow_vars_ops, q_values  = create_train_op(
    state_ph, next_state_ph, action_ph, reward_ph,
    is_not_final_ph, is_training_ph, hparams)

# episode counter
episodes = tf.Variable(0.0, trainable=False, name='episodes')
counter_op = episodes.assign_add(1)

# initialize session
sess = tf.Session()
sess.run(tf.global_variables_initializer())

total_steps = 0
experience = deque(maxlen=int(hparams.replay_buffer_size))

epsilon = hparams.epsilon_start
epsilon_linear_step = (
    hparams.epsilon_start - hparams.epsilon_linear_end) / hparams.num_epsilon_linear_steps

# Create saver
saver = tf.train.Saver()
steps = []
epsilons = []
rewards = []
avg_rewards = []
episodes_list = []

env = gym.make('LunarLander-v2')

for ep in range(hparams.max_num_episodes):
  total_reward = 0
  steps_in_ep = 0

  observation = env.reset()

  for t in range(hparams.max_steps_per_episode):

      if np.random.random() < epsilon:
          action = np.random.randint(hparams.num_actions)
      else:
          q_s = sess.run(q_values,
                         feed_dict={state_ph: observation[None], is_training_ph: False})
          action = np.argmax(q_s)

      next_observation, reward, done, _info = env.step(action)
      total_reward += reward

      experience.append((observation, action, reward, next_observation,
                         0.0 if done else 1.0))

      if total_steps % hparams.update_slow_target_every_steps == 0:
          sess.run(update_slow_vars_ops)

      if len(experience) >= hparams.batch_size:
          minibatch = random.sample(experience, hparams.batch_size)

          sess.run(train_op,
                       feed_dict={
                           state_ph: np.asarray([elem[0] for elem in minibatch]),
                           action_ph: np.asarray([elem[1] for elem in minibatch]),
                           reward_ph: np.asarray([elem[2] for elem in minibatch]),
                           next_state_ph: np.asarray([elem[3] for elem in minibatch]),
                           is_not_final_ph: np.asarray([elem[4] for elem in minibatch]),
                           is_training_ph: True})

      observation = next_observation
      total_steps += 1
      steps_in_ep += 1

      # linearly decay epsilon
      if total_steps < hparams.num_epsilon_linear_steps:
          epsilon -= epsilon_linear_step
      # exponentially decay epsilon
      elif done:
          epsilon *= hparams.epsilon_exp_decay

      if done:
          # Increment episode counter
          _ = sess.run(counter_op)
          break


env.close()
