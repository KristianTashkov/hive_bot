import numpy as np
import tensorflow as tf
from itertools import product

from engine.actions import create_all_actions


def ceildiv(x, y):
    return tf.cast(tf.ceil(tf.truediv(x, y)), tf.int32)


class Model:
    def __init__(self, game, player_id):
        self.player_id = player_id
        self.all_actions = create_all_actions(game, player_id)

    def __enter__(self):
        self.setup_predicting_graph()
        self.setup_training_graph()

        self.session = tf.Session(
            config=tf.ConfigProto(allow_soft_placement=True,
                                  log_device_placement=True))
        self.session.run(tf.global_variables_initializer())

        self.gradBuffer = self.session.run(tf.trainable_variables())
        for ix, grad in enumerate(self.gradBuffer):
            self.gradBuffer[ix] = grad * 0
        return self

    def setup_predicting_graph(self):
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

        self.input_tensor = tf.placeholder(tf.float32, (None, 22, 22, 60), name='state')
        self.allowed_actions_tensor = tf.placeholder(tf.float32, (None, len(self.all_actions)), name='allowed_actions')

        h, sh = self.input_tensor, (22, 22)
        h, sh = conv(h, sh, 16, maxpool=True)
        h, sh = conv(h, sh, 32, maxpool=True)
        h, sh = conv(h, sh, 48, maxpool=False)
        features = tf.contrib.layers.flatten(h)

        self.logits = tf.layers.dense(features, len(self.all_actions), activation=tf.nn.leaky_relu)
        self.output = tf.nn.softmax(self.logits)
        self.output = self.output * self.allowed_actions_tensor

    def setup_training_graph(self):
        self.reward_holder = tf.placeholder(shape=[None], dtype=tf.float32)
        self.action_holder = tf.placeholder(shape=[None], dtype=tf.int32)
        self.indexes = tf.range(0, tf.shape(self.output)[0]) * tf.shape(self.output)[1] + self.action_holder
        self.responsible_outputs = tf.gather(tf.reshape(self.output, [-1]), self.indexes)

        self.loss = -tf.reduce_mean(tf.log(self.responsible_outputs) * self.reward_holder)
        tvars = tf.trainable_variables()
        self.gradient_holders = []
        for idx, var in enumerate(tvars):
            placeholder = tf.placeholder(tf.float32, name=str(idx) + '_holder')
            self.gradient_holders.append(placeholder)

        self.gradients = tf.gradients(self.loss, tvars)
        optimizer = tf.train.AdamOptimizer(learning_rate=1e-3)
        self.update_batch = optimizer.apply_gradients(zip(self.gradient_holders, tvars))

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.session.close()

    def piece_embedding(self, piece):
        embedding = np.full((12, ), 0, dtype=np.float32)
        if piece is None:
            return embedding
        if piece.color == self.player_id:
            embedding[0] = 1.0
        embedding[1 + piece.id] = 1.0
        return embedding

    def get_state(self, hive_game):
        positions = [x.position for x in hive_game.all_pieces()]
        board_state = np.full((22, 22, 60), 0)
        allowed_actions = np.array([x.can_be_played() for x in self.all_actions], dtype=np.float32)
        if len(positions) == 0:
            return {'board': board_state,
                    'allowed_actions': allowed_actions}
        min_x, min_y = np.min([x[0] for x in positions]), np.min([x[1] for x in positions])
        max_x, max_y = np.max([x[0] for x in positions]), np.max([x[1] for x in positions])
        for x, y in product(range(min_x, max_x + 1), range(min_y, max_y + 1)):
            stack = hive_game.get_stack((x, y))
            normalized_x = x - min_x
            normalized_y = y - min_y
            for i in range(5):
                piece = stack[i] if i < len(stack) else None
                board_state[normalized_x, normalized_y, i * 12: (i + 1) * 12] = self.piece_embedding(piece)

        return {'board': board_state,
                'allowed_actions': allowed_actions}

    def choose_action(self, state, is_training):
        if np.sum(state['allowed_actions']) == 0:
            return None
        if not is_training or np.random.random() < 0.8:
            output = self.session.run(
                self.output,
                feed_dict={self.input_tensor: state['board'][np.newaxis, ...],
                           self.allowed_actions_tensor: state['allowed_actions'][np.newaxis, ...]})
            normalized_output = output[0] / np.sum(output[0])
            if not is_training:
                action_id = np.argmax(normalized_output)
            else:
                action_id = np.random.choice(np.arange(len(self.all_actions)),
                                             p=normalized_output)

        else:
            action_id = np.random.choice(np.arange(len(self.all_actions)),
                                         p=state['allowed_actions'] / np.sum(state['allowed_actions']))
        return action_id, self.all_actions[action_id]

    def propagate_reward(self, state, chosen_action_id, reward):
        pass
