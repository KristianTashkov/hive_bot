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

    def _play_move(self, game):
        common_data = {}
        available_actions = [x for x in self.all_actions
                             if x not in self.done_actions and x.can_be_played(game, common_data)]
        if len(available_actions) == 0 and len(self.done_actions) > 0:
            game.to_play(list(self.done_actions)[0])
            return False
        action = np.random.choice(available_actions, 1)[0] if len(available_actions) > 0 else None

        game.play_action(action)
        self.done_actions.add(action)
        return True

    def play_move(self, game):
        player_id = game.to_play
        self.done_actions = set()
        self._play_move(game)
        while game.get_winner() == (player_id + 1) % 2:
            game.game_history[-1].copy_to(game)
            if not self._play_move(game):
                return

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        pass
