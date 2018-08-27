import numpy as np

from engine.actions import create_all_actions
from zoo.model import ConvModel


class ModelPlayer:
    def __init__(self, *args, model_cls=ConvModel, **kwargs):
        self.model = model_cls(*args, **kwargs)

    def __enter__(self):
        self.model.__enter__()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.model.__exit__(exc_type, exc_val, exc_tb)

    def play_move(self, game):
        state = self.model.get_state(game)
        _, action = self.model.choose_action(state)
        game.play_action(action)


class RandomPlayer:
    def __init__(self):
        self.all_actions = create_all_actions()

    def play_move(self, game):
        common_data = {}
        available_actions = [x for x in self.all_actions if x.can_be_played(game, common_data)]
        action = np.random.choice(available_actions, 1)[0] if len(available_actions) > 0 else None
        game.play_action(action)

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        pass
