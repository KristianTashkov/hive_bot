import numpy as np
from itertools import product

from hive_game import HiveGame
from game_piece import GamePieceType, GamePiece


class Action:
    def __init__(self, hive_game):
        self.game = hive_game


class DeployAction(Action):
    def __init__(self, hive_game, color, piece_type, position):
        super().__init__(hive_game)
        self.color = color
        self.piece_type = piece_type
        self.position = position

    def can_be_played(self):
        # Wrong turn
        if self.color != self.game.to_play:
            return False
        player_pieces = self.game.player_pieces(self.color)
        enemy_pieces = self.game.player_pieces((self.color + 1) % 2)

        # No queen on 4th turn
        if (self.game.queens[self.color] is None and self.piece_type != GamePieceType.QUEEN_BEE
                and len(player_pieces) == 3):
            return False
        has_pieces_left = len([x for x in player_pieces
                               if x.piece_type == self.piece_type]) < GamePieceType.PIECE_COUNT[self.piece_type]

        # Has piece left and position is empty
        if (not has_pieces_left) or (self.game.get_top_piece(self.position) is not None):
            return False
        neighbor_pieces_colors = {x.color for x in self.game.neighbor_pieces(self.position)}

        # Doesn't touch enemies and touches ally
        if len(player_pieces) != 0 and neighbor_pieces_colors != {self.color}:
            return False

        # First two positions are (0, 0) and (1, 0)
        if len(player_pieces) == 0:
            if self.position[1] != 0 or self.position[0] != len(enemy_pieces):
                return False
        return True

    def activate(self):
        new_piece = GamePiece(
            self.game, self.color, self.piece_type, (self.position[0], self.position[1]))
        self.game.set_top_piece(self.position, new_piece)
        if self.piece_type == GamePieceType.QUEEN_BEE:
            self.game.queens[self.color] = new_piece

    def __repr__(self):
        return '{}[{}, {}]'.format(self.piece_type, self.position[0], self.position[1])


class MoveAction(Action):
    def __init__(self, hive_game, start_position, end_position):
        super().__init__(hive_game)
        self.start_position = start_position
        self.end_position = end_position

    def _set_dfs_state(self, speed, moving_piece):
        all_positions = np.array([x.position for x in self.game.all_pieces()])
        self.min_x, self.max_x = np.min(all_positions[:, 0]) - 1, np.max(all_positions[:, 0]) + 1
        self.min_y, self.max_y = np.min(all_positions[:, 1]) - 1, np.max(all_positions[:, 1]) + 1
        self.dfs_found = False
        self.speed = speed
        self.moving_piece = moving_piece

    def _dfs(self, current, level, used):
        if current == self.end_position:
            if self.speed == np.inf or self.speed == level:
                self.dfs_found = True
                return
        if level >= self.speed:
            return

        current_neighbors_positions = {x.position for x in self.game.neighbor_pieces(current)
                                       if x != self.moving_piece}
        for dx, dy in HiveGame.NEIGHBORS_DIRECTION:
            new_x, new_y = current[0] + dx, current[1] + dy
            if new_x < self.min_x or new_x > self.max_x or new_y < self.min_y or new_y > self.max_y:
                continue
            if self.game.get_top_piece((new_x, new_y)) is not None:
                continue
            neighbor_pieces = [x for x in self.game.neighbor_pieces((new_x, new_y)) if x != self.moving_piece]
            if len(neighbor_pieces) == 0:
                continue
            if (new_x, new_y) in used:
                continue
            occupied_neighbor_positions = {x.position for x in neighbor_pieces}

            # Can't go through tiny hole
            if len(current_neighbors_positions.intersection(occupied_neighbor_positions)) == 2:
                continue

            new_used = used.copy()
            new_used.update({(new_x, new_y)})
            self._dfs((new_x, new_y), level + 1, new_used)
            if self.dfs_found:
                return

    def _slide_movement_allowed(self, speed, moving_piece):
        self._set_dfs_state(speed, moving_piece)
        self._dfs(self.start_position, 0, {self.start_position})
        return self.dfs_found

    def can_be_played(self):
        moving_piece = self.game.get_top_piece(self.start_position)

        # Same start and end not allowed
        if self.start_position == self.end_position:
            return False

        # Moving piece is current players
        if moving_piece is None or moving_piece.color != self.game.to_play:
            return False

        # Moving player has queen deployed
        if self.game.queens[moving_piece.color] is None:
            return False

        # The piece is non-connecting
        if self.game.is_connecting_piece(moving_piece):
            return False

        # Position is empty unless beetle
        if moving_piece.piece_type != GamePieceType.BEETLE and self.game.get_top_piece(self.end_position) is not None:
            return False

        # Is movement valid by type
        if moving_piece.piece_type == GamePieceType.SOLDIER_ANT:
            return self._slide_movement_allowed(np.inf, moving_piece)
        elif moving_piece.piece_type == GamePieceType.SPIDER:
            return self._slide_movement_allowed(3, moving_piece)
        elif moving_piece.piece_type == GamePieceType.QUEEN_BEE:
            return self._slide_movement_allowed(1, moving_piece)
        elif moving_piece.piece_type == GamePieceType.BEETLE:
            # one step movement
            allowed_positions = [(self.start_position[0] + dx, self.start_position[1] + dy)
                                 for dx, dy in HiveGame.NEIGHBORS_DIRECTION]
            return self.end_position in allowed_positions
        elif moving_piece.piece_type == GamePieceType.GRASSHOPPER:
            dx, dy = (self.end_position[0] - self.start_position[0]), (self.end_position[1] - self.start_position[1])
            # goes in a single direction
            if abs(dx) > 0 and abs(dy) > 0 and -dx != dy:
                return False
            x_step = dx / abs(dx) if dx != 0 else 0
            y_step = dy / abs(dy) if dy != 0 else 0
            # jumps over at least one piece
            if abs(x_step) == 1 or abs(y_step) == 1:
                return False

            # all pieces jumped over are filled
            for step in range(1, np.max(np.abs([dx, dy]))):
                if self.game.get_top_piece((self.start_position[0] + step * x_step,
                                            self.start_position[1] + step * y_step)) is None:
                    return False
            return True
        else:
            raise ValueError('Invalid game piece')

    def __repr__(self):
        return '[{} -> [{}, {}]]'.format(self.moving_piece,
                                         self.end_position[0], self.end_position[1])

    def activate(self):
        piece = self.game.get_top_piece(self.start_position)
        self.game.drop_top_piece(self.start_position)
        self.game.set_top_piece(self.end_position, piece)
        piece.position = self.end_position


def all_actions(game):
    available_actions = []
    for piece_type in GamePieceType.PIECE_COUNT.keys():
        for x, y in product(range(-5, 5), range(-5, 5)):
            deploy = DeployAction(game, game.to_play, piece_type, (x, y))
            if deploy.can_be_played():
                available_actions.append(deploy)
    for start_x, start_y in product(range(-5, 5), range(-5, 5)):
        for end_x, end_y in product(range(-5, 5), range(-5, 5)):
            move = MoveAction(game, (start_x, start_y), (end_x, end_y))
            if move.can_be_played():
                available_actions.append(move)
    if len(available_actions) == 0:
        return [None]
    return available_actions