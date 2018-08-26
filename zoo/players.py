import numpy as np

from engine.actions import create_all_actions
from zoo.model import Model


class ModelPlayer(Model):
    def play_move(self):
        state = self.get_state(self.game)
        _, action = self.choose_action(state)
        self.game.play_action(action)


class RandomPlayer:
    def __init__(self, game, player_id):
        self.game = game
        self.player_id = player_id
        self.all_actions = create_all_actions(game, player_id)

    def play_move(self):
        common_data = {}
        available_actions = [x for x in self.all_actions if x.can_be_played(common_data)]
        action = np.random.choice(available_actions, 1)[0] if len(available_actions) > 0 else None
        self.game.play_action(action)

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        pass
