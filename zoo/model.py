import os
import numpy as np
import tensorflow as tf
from itertools import product

from engine.actions import create_all_actions
from engine.game_piece import GamePieceType


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
    def __init__(self, is_training=False, checkpoint=None, save_dir=None):
        self.is_training = is_training
        self.checkpoint = checkpoint
        self.save_dir = save_dir if save_dir is not None else 'D:\\code\\hive\\checkpoints\\'
        self.all_actions = create_all_actions()

    def __enter__(self):
        self.model_graph = tf.Graph()
        with self.model_graph.as_default():
            self.allowed_actions_tensor = tf.placeholder(tf.float32, (None, len(self.all_actions)),
                                                         name='allowed_actions')
            self.input_tensor, self.output = self.setup_predicting_graph()
            self.setup_training_graph()

        self.session = tf.Session(
            graph=self.model_graph,
            config=tf.ConfigProto(allow_soft_placement=True))
        with self.session.as_default():
            with self.model_graph.as_default():
                self.session.run(tf.global_variables_initializer())

                self.gradBuffer = self.session.run(tf.trainable_variables())
                for ix, grad in enumerate(self.gradBuffer):
                    self.gradBuffer[ix] = grad * 0
                if self.checkpoint is not None:
                    self.load_checkpoint()
        return self

    def setup_predicting_graph(self):
        raise NotImplemented()

    def get_board_state(self, hive_game):
        raise NotImplemented()

    def setup_training_graph(self):
        self.reward_holder = tf.placeholder(shape=[None], dtype=tf.float32)
        self.action_holder = tf.placeholder(shape=[None], dtype=tf.int32)
        self.indexes = tf.range(0, tf.shape(self.output)[0]) * tf.shape(self.output)[1] + self.action_holder
        self.responsible_outputs = tf.gather(tf.reshape(self.output, [-1]), self.indexes)

        self.loss = -tf.reduce_mean(tf.log(self.responsible_outputs) * self.reward_holder)
        self.loss = tf.Print(self.loss, [self.loss, tf.reduce_mean(self.responsible_outputs) * 100], "loss: ")
        tvars = tf.trainable_variables()
        self.gradient_holders = []
        for idx, var in enumerate(tvars):
            placeholder = tf.placeholder(tf.float32, name=str(idx) + '_holder')
            self.gradient_holders.append(placeholder)

        self.gradients = [x for x in tf.gradients(self.loss, tvars)]
        optimizer = tf.train.AdamOptimizer(learning_rate=0.0001)
        self.update_batch = optimizer.apply_gradients(zip(self.gradient_holders, tvars))

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.session.close()

    def get_state(self, hive_game):
        common_data = {}
        allowed_actions = np.array([x.can_be_played(hive_game, common_data)
                                    for x in self.all_actions], dtype=np.float32)
        return {'board': self.get_board_state(hive_game)[np.newaxis, ...],
                'allowed_actions': allowed_actions[np.newaxis, ...]}

    def choose_action(self, state):
        if np.sum(state['allowed_actions'][0]) == 0:
            return -1, None
        if not self.is_training or np.random.random() < 0.8:
            with self.session.as_default():
                with self.model_graph.as_default():
                    output = self.session.run(
                        self.output,
                        feed_dict={self.input_tensor: state['board'],
                                   self.allowed_actions_tensor: state['allowed_actions']})
            normalized_output = output[0].copy()
            normalized_output[state['allowed_actions'][0] == 0] = 0
            print("output", [round(x, 2) for x in np.array(sorted(normalized_output * -1)[:3]) * -100], end='\r')
            if not self.is_training:
                action_id = np.argmax(normalized_output)
            else:
                normalized_output = output[0]
                normalized_output[state['allowed_actions'][0] == 0] = 0
                normalized_output /= np.sum(normalized_output)
                action_id = np.random.choice(np.arange(len(self.all_actions)),
                                             p=normalized_output)

            if state['allowed_actions'][0][action_id] != 1:
                print(output)
                raise KeyboardInterrupt()

        else:
            action_id = np.random.choice(np.arange(len(self.all_actions)),
                                         p=state['allowed_actions'][0] / np.sum(state['allowed_actions'][0]))
        return action_id, self.all_actions[action_id]

    def propagate_reward(self, state, all_allowed, played_actions, reward):
        with self.session.as_default():
            with self.model_graph.as_default():
                grads, _ = self.session.run(
                    [self.gradients, self.loss],
                    feed_dict={self.input_tensor: state.astype(np.float32),
                               self.allowed_actions_tensor: all_allowed.astype(np.float32),
                               self.action_holder: played_actions.astype(np.float32),
                               self.reward_holder: reward})
                for idx, grad in enumerate(grads):
                    self.gradBuffer[idx] += grad
                feed_dict = dict(zip(self.gradient_holders, self.gradBuffer))
                _ = self.session.run(self.update_batch, feed_dict=feed_dict)
                for ix, grad in enumerate(self.gradBuffer):
                    self.gradBuffer[ix] = grad * 0

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
    def setup_predicting_graph(self):
        input_tensor = tf.placeholder(tf.float32, (None, 22, 22, 110), name='state')

        h, sh = input_tensor, (22, 22)
        h, sh = conv(h, sh, 16, maxpool=True)
        h, sh = conv(h, sh, 32, maxpool=False)
        features = tf.contrib.layers.flatten(h)

        features = tf.layers.dense(features, 2048, activation=tf.nn.leaky_relu)
        logits = tf.layers.dense(features, len(self.all_actions))
        logits *= self.allowed_actions_tensor
        output = tf.nn.softmax(logits, axis=-1) * 0.95 + 0.05 / len(self.all_actions)
        return input_tensor, output

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


