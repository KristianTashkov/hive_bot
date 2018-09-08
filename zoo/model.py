import os
from collections import defaultdict

import numpy as np
import tensorflow as tf
from itertools import product


def ceildiv(x, y):
    return tf.cast(tf.ceil(tf.truediv(x, y)), tf.int32)


def conv(inputs, shapes, nunits, maxpool=False):
    outputs = tf.layers.conv2d(inputs, nunits, (3, 3),
                               activation=None,
                               padding='same',
                               use_bias=False)
    outputs = tf.nn.leaky_relu(outputs)
    if maxpool:
        outputs = tf.layers.max_pooling2d(outputs, (2, 2), 2,
                                          padding='same')
        shapes = ceildiv(shapes, (2, 2))
    return outputs, shapes


class Model:
    def __init__(self, all_actions, is_training=False, checkpoint=None, save_dir=None):
        self.is_training = is_training
        self.checkpoint = checkpoint
        self.save_dir = save_dir if save_dir is not None else 'D:\\code\\hive\\checkpoints\\'
        self.all_actions = all_actions

    def __enter__(self):
        self.model_graph = tf.Graph()
        with self.model_graph.as_default():
            self.global_step = tf.Variable(0, trainable=False, name='global_step')
            self.allowed_actions_tensor = tf.placeholder(tf.float32, (None, len(self.all_actions[0])),
                                                         name='allowed_actions')
            self.state_input, self.actor_output, self.critic_output = self.setup_graph()
            self.setup_training_graph()

        self.session = tf.Session(
            graph=self.model_graph,
            config=tf.ConfigProto(allow_soft_placement=True))
        with self.session.as_default():
            with self.model_graph.as_default():
                self.session.run(tf.global_variables_initializer())

                if self.checkpoint is not None:
                    self.load_checkpoint()
        return self

    def setup_graph(self):
        raise NotImplemented()

    def get_board_state(self, hive_game):
        raise NotImplemented()

    def setup_training_graph(self):
        self.advantage_holder = tf.placeholder(shape=[None], dtype=tf.float32)
        self.reward_holder = tf.placeholder(shape=[None], dtype=tf.float32)
        self.action_holder = tf.placeholder(shape=[None], dtype=tf.int32)
        self.indexes = tf.range(0, tf.shape(self.actor_output)[0]) * tf.shape(self.actor_output)[1] + self.action_holder
        self.responsible_outputs = tf.gather(tf.reshape(self.actor_output, [-1]), self.indexes)
        self.responsible_outputs = tf.where(
            self.reward_holder > 0,
            tf.clip_by_value(self.responsible_outputs, 0.0, 0.95),
            tf.clip_by_value(self.responsible_outputs, 0.05 / len(self.all_actions[0]), 1))

        rewards = tf.log(self.responsible_outputs) * self.advantage_holder
        policy_loss = -tf.reduce_mean(rewards)
        critic_loss = tf.losses.mean_squared_error(self.reward_holder, self.critic_output)
        entropy_loss = 0.01 * tf.reduce_mean(-tf.reduce_sum(tf.log(self.actor_output) * self.actor_output, axis=-1))

        loss = policy_loss + critic_loss - entropy_loss
        gradients = tf.gradients(loss, tf.trainable_variables())
        gradients, global_norm = tf.clip_by_global_norm(gradients, 0.5)
        optimizer = tf.train.AdamOptimizer(learning_rate=0.001)

        self.global_norm = tf.Print(
            global_norm, [tf.reduce_sum(tf.abs(rewards)), critic_loss, entropy_loss, global_norm], 'info: ')
        self.train_op = optimizer.apply_gradients(zip(gradients, tf.trainable_variables()),
                                                  global_step=self.global_step)

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.session.close()

    def get_state(self, hive_game):
        common_data = {}
        allowed_actions = np.array([x.can_be_played(hive_game, common_data)
                                    for x in self.all_actions[hive_game.to_play]], dtype=np.float32)
        return {'board': self.get_board_state(hive_game)[np.newaxis, ...],
                'allowed_actions': allowed_actions[np.newaxis, ...],
                'to_play': hive_game.to_play}

    def choose_action(self, state):
        uniform_representations = [(index, x.uniform_representation())
                                   for index, x in enumerate(self.all_actions[state['to_play']])
                                   if state['allowed_actions'][0][index] == 1]
        groups = defaultdict(list)
        for index, representation in uniform_representations:
            groups[representation].append(index)
        groups = list(groups.values())

        with self.session.as_default():
            with self.model_graph.as_default():
                actor_output, critic_output = self.session.run(
                    [self.actor_output, self.critic_output],
                    feed_dict={self.state_input: state['board'],
                               self.allowed_actions_tensor: state['allowed_actions']})
        normalized_output = actor_output[0].copy()
        normalized_output[state['allowed_actions'][0] == 0] = 0
        group_scores = np.array([np.sum([normalized_output[index] for index in group]) for group in groups])
        group_scores /= np.sum(group_scores)

        if not self.is_training:
            group_id = np.argmax(group_scores)
        else:
            group_id = np.random.choice(np.arange(len(groups)), p=group_scores)
        action_id = np.random.choice(groups[group_id], 1)[0]

        if state['allowed_actions'][0][action_id] != 1:
            print(actor_output)
            raise KeyboardInterrupt()

        print("output", round(critic_output, 2), [round(x, 2) for x in np.array(sorted(group_scores * -1)[:3]) * -100], end='\r')
        return critic_output, action_id, self.all_actions[state['to_play']][action_id]

    def train_model(self, state, all_allowed, played_actions, reward, advantage):
        with self.session.as_default():
            with self.model_graph.as_default():
                _, _ = self.session.run(
                    [self.train_op, self.global_norm],
                    feed_dict={self.state_input: state,
                               self.allowed_actions_tensor: all_allowed,
                               self.action_holder: played_actions.astype(np.float32),
                               self.reward_holder: reward,
                               self.advantage_holder: advantage})

    def save(self, name, step):
        with self.session.as_default():
            with self.model_graph.as_default():
                saver = tf.train.Saver(tf.trainable_variables())
                directory = os.path.join(self.save_dir, name)
                if not os.path.exists(directory):
                    os.mkdir(directory)
                saver.save(self.session, os.path.join(directory, 'model.ckpt'), global_step=step)

    def load_checkpoint(self):
        with self.session.as_default():
            with self.model_graph.as_default():
                saver = tf.train.Saver(tf.trainable_variables())
                saver.restore(self.session, self.checkpoint)


class ConvModel(Model):
    def setup_graph(self):
        input_tensor = tf.placeholder(tf.float32, (None, 22, 22, 110), name='state')

        h, sh = input_tensor, (22, 22)
        h, sh = conv(h, sh, 16, maxpool=True)
        h, sh = conv(h, sh, 32, maxpool=False)
        features = tf.contrib.layers.flatten(h)

        logits = tf.layers.dense(features, len(self.all_actions[0]))
        logits *= self.allowed_actions_tensor
        actor_output = tf.nn.softmax(logits, axis=-1)

        critic_output = tf.squeeze(tf.layers.dense(features, 1))

        return input_tensor, actor_output, critic_output

    def piece_embedding(self, piece, player_id):
        embedding = np.full((22, ), 0, dtype=np.float32)
        if piece is None:
            return embedding
        offset = 0 if piece.color == player_id else 11
        embedding[offset + piece.id] = 1.0
        return embedding

    def get_board_state(self, hive_game):
        positions = [x.position for x in hive_game.all_pieces()]
        board_state = np.full((22, 22, 110), 0)
        if len(positions) == 0:
            return board_state
        min_x, min_y = np.min([x[0] for x in positions]), np.min([x[1] for x in positions])
        max_x, max_y = np.max([x[0] for x in positions]), np.max([x[1] for x in positions])
        for x, y in product(range(min_x, max_x + 1), range(min_y, max_y + 1)):
            stack = hive_game.get_stack((x, y))
            normalized_x = x - min_x
            normalized_y = y - min_y
            for i in range(5):
                piece = stack[i] if i < len(stack) else None
                board_state[normalized_x, normalized_y,
                            i * 22: (i + 1) * 22] = self.piece_embedding(piece, hive_game.to_play)
        return board_state


