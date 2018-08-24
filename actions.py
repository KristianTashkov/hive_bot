import numpy as np
from itertools import product

from hive_game import HiveGame
from game_piece import GamePieceType


def target_position(start_position, relative_direction):
    return (start_position[0] + relative_direction[0],
            start_position[1] + relative_direction[1])


def hex_distance(a, b):
    return (abs(a[0] - b[0]) + abs(a[0] + a[1] - b[0] - b[1]) + abs(a[1] - b[1])) / 2


class Action:
    def __init__(self, hive_game):
        self.game = hive_game


class InitialDeployAction(Action):
    def __init__(self, hive_game, color, piece_type):
        super().__init__(hive_game)
        self.color = color
        self.piece_type = piece_type

    def can_be_played(self):
        # Wrong turn
        if self.color != self.game.to_play:
            return False

        player_pieces = self.game.player_pieces(self.color)
        if len(player_pieces) != 0:
            return False

        return True

    def activate(self):
        enemy_pieces = self.game.player_pieces((self.color + 1) % 2)
        position = (0, 0) if len(enemy_pieces) == 0 else (1, 0)
        self.game.deploy_piece(position, self.piece_type)

    def __repr__(self):
        return 'Initial {}'.format(self.piece_type)


class DeployAction(Action):
    def __init__(self, hive_game, color, piece_type, next_to_id, relative_direction):
        super().__init__(hive_game)
        self.color = color
        self.piece_type = piece_type
        self.next_to_id = next_to_id
        self.relative_direction = relative_direction

    def _get_real_position(self):
        neighbor = self.game.get_piece(self.color, self.next_to_id)
        return target_position(neighbor.position, self.relative_direction)

    def can_be_played(self):
        # Wrong turn
        if self.color != self.game.to_play:
            return False

        neighbor = self.game.get_piece(self.color, self.next_to_id)
        if neighbor is None:
            return False

        player_pieces = self.game.player_pieces(self.color)

        # No queen on 4th turn
        if (self.game.get_piece(self.color, 0) is None and self.piece_type != GamePieceType.QUEEN_BEE
                and len(player_pieces) == 3):
            return False
        has_pieces_left = len([x for x in player_pieces
                               if x.piece_type == self.piece_type]) < GamePieceType.PIECE_COUNT[self.piece_type]

        # Has piece left and position is empty
        position = self._get_real_position()
        if (not has_pieces_left) or (self.game.get_top_piece(position) is not None):
            return False
        neighbor_pieces_colors = {x.color for x in self.game.neighbor_pieces(position)}

        # Doesn't touch enemies
        if (self.color + 1) % 2 in neighbor_pieces_colors:
            return False

        return True

    def activate(self):
        position = self._get_real_position()
        self.game.deploy_piece(position, self.piece_type)

    def __repr__(self):
        position = self._get_real_position()
        return '{}[{}, {}]'.format(self.piece_type, position[0], position[1])


class BaseMove(Action):
    def __init__(self, hive_game, color, piece_id):
        super().__init__(hive_game)
        self.color = color
        self.piece_id = piece_id

    def end_position(self):
        raise NotImplemented()

    def moving_piece(self):
        return self.game.get_piece(self.color, self.piece_id)

    def can_be_played(self):
        # Wrong turn
        if self.color != self.game.to_play:
            return False

        # Piece is deployed
        piece = self.moving_piece()
        if piece is None:
            return False

        # Piece is on top
        if not piece.is_on_top():
            return False

        # Moving player has queen deployed
        if self.game.get_piece(self.color, 0) is None:
            return False

        # The piece is non-connecting
        if self.game.is_connecting_piece(piece):
            return False

        return True

    def activate(self):
        piece = self.moving_piece()
        self.game.drop_top_piece(piece.position)
        end_position = self.end_position()
        self.game.set_top_piece(end_position, piece)
        piece.position = end_position


class GrasshopperMove(BaseMove):
    def __init__(self, hive_game, color, piece_id, relative_direction):
        super().__init__(hive_game, color, piece_id)
        self.relative_direction = relative_direction

    def _simulate_jump(self, piece):
        jumped_over = []
        current_position = piece.position
        while True:
            current_position = target_position(current_position, self.relative_direction)
            jumped_over_piece = self.game.get_top_piece(current_position)
            if jumped_over_piece is None:
                break
            jumped_over.append(jumped_over_piece)
        return current_position, jumped_over

    def can_be_played(self):
        if not super().can_be_played():
            return False

        piece = self.moving_piece()
        jump_position, jumped_over = self._simulate_jump(piece)
        if len(jumped_over) == 0:
            return False
        return True

    def end_position(self):
        piece = self.moving_piece()
        return self._simulate_jump(piece)[0]

    def __repr__(self):
        return '[Grasshopper{} -> [{}, {}]]'.format(
            self.piece_id, self.relative_direction[0], self.relative_direction[1])


class BeetleMove(BaseMove):
    def __init__(self, hive_game, color, piece_id, relative_direction):
        super().__init__(hive_game, color, piece_id)
        self.relative_direction = relative_direction

    def can_be_played(self):
        if not super().can_be_played():
            return False
        piece = self.moving_piece()

        new_position = target_position(piece.position, self.relative_direction)
        neighbor_pieces = [x for x in self.game.neighbor_pieces(new_position) if x != piece]
        # No running away from hive
        if len(neighbor_pieces) == 0:
            return False

        # Freedom to move rule for beetles
        beetle_stack_size = len(self.game.get_stack(piece.position)) - 1
        target_stack_size = len(self.game.get_stack(new_position))
        index_direction = HiveGame.NEIGHBORS_DIRECTION.index(self.relative_direction)
        prev_direction = HiveGame.NEIGHBORS_DIRECTION[(index_direction - 1) % len(HiveGame.NEIGHBORS_DIRECTION)]
        next_direction = HiveGame.NEIGHBORS_DIRECTION[(index_direction + 1) % len(HiveGame.NEIGHBORS_DIRECTION)]
        gate1_size = len(self.game.get_stack(target_position(piece.position, prev_direction)))
        gate2_size = len(self.game.get_stack(target_position(piece.position, next_direction)))
        if (beetle_stack_size < gate1_size and beetle_stack_size < gate2_size and
                target_stack_size < gate1_size and target_stack_size < gate2_size):
            return False
        return True

    def end_position(self):
        piece = self.moving_piece()
        return target_position(piece.position, self.relative_direction)

    def __repr__(self):
        return '[Beetle{} -> [{}, {}]]'.format(
            self.piece_id, self.relative_direction[0], self.relative_direction[1])


class QueenMove(BeetleMove):
    def __init__(self, hive_game, color, piece_id, relative_direction):
        super().__init__(hive_game, color, piece_id, relative_direction)

    def can_be_played(self):
        if not super().can_be_played():
            return False

        # Can't go on top like beetle
        piece = self.moving_piece()
        new_position = target_position(piece.position, self.relative_direction)
        if self.game.get_top_piece(new_position) is not None:
            return False

        return True

    def __repr__(self):
        return '[Queen{} -> [{}, {}]]'.format(
            self.piece_id, self.relative_direction[0], self.relative_direction[1])


class MoveAction(BaseMove):
    def __init__(self, hive_game, color, piece_id, speed):
        super().__init__(hive_game, color, piece_id)
        self.speed = speed

    def _set_dfs_state(self):
        all_positions = np.array([x.position for x in self.game.all_pieces()])
        self.min_x, self.max_x = np.min(all_positions[:, 0]) - 1, np.max(all_positions[:, 0]) + 1
        self.min_y, self.max_y = np.min(all_positions[:, 1]) - 1, np.max(all_positions[:, 1]) + 1

    def _dfs(self, current, level, used):
        if current == self.end_position():
            if self.speed == np.inf or self.speed == level:
                return True
        if level >= self.speed:
            return False

        current_neighbors = {x for x in self.game.neighbor_pieces(current)}
        for index_direction, (dx, dy) in enumerate(HiveGame.NEIGHBORS_DIRECTION):
            new_x, new_y = current[0] + dx, current[1] + dy
            if new_x < self.min_x or new_x > self.max_x or new_y < self.min_y or new_y > self.max_y:
                continue
            if self.game.get_top_piece((new_x, new_y)) is not None:
                continue
            neighbor_pieces = {x for x in self.game.neighbor_pieces((new_x, new_y)) if x != self.moving_piece()}
            if len(neighbor_pieces) == 0 or len(current_neighbors.intersection(neighbor_pieces)) == 0:
                continue
            if (new_x, new_y) in used:
                continue

            prev_direction = HiveGame.NEIGHBORS_DIRECTION[(index_direction - 1) % len(HiveGame.NEIGHBORS_DIRECTION)]
            next_direction = HiveGame.NEIGHBORS_DIRECTION[(index_direction + 1) % len(HiveGame.NEIGHBORS_DIRECTION)]
            if (self.game.get_top_piece(target_position(current, prev_direction)) is not None and
                    self.game.get_top_piece(target_position(current, next_direction)) is not None):
                continue

            new_used = used.copy()
            new_used.update({(new_x, new_y)})
            if self._dfs((new_x, new_y), level + 1, new_used):
                return True
        return False

    def can_be_played(self):
        if not super().can_be_played():
            return False

        moving_piece = self.moving_piece()
        # Same start and end not allowed
        if moving_piece.position == self.end_position():
            return False

        # Position is not empty
        if self.game.get_top_piece(self.end_position()) is not None:
            return False

        self._set_dfs_state()
        return self._dfs(moving_piece.position, 0, {moving_piece.position})


class SpiderMove(MoveAction):
    def __init__(self, hive_game, color, piece_id, relative_direction):
        super().__init__(hive_game, color, piece_id, 3)
        self.relative_direction = relative_direction

    def end_position(self):
        piece = self.moving_piece()
        return target_position(piece.position, self.relative_direction)

    def __repr__(self):
        return '[Spider{} -> [{}, {}]]'.format(
            self.piece_id, self.relative_direction[0], self.relative_direction[1])


class AntMove(MoveAction):
    def __init__(self, hive_game, color, piece_id, next_to_id, next_to_color, relative_direction):
        super().__init__(hive_game, color, piece_id, np.inf)
        self.next_to_id = next_to_id
        self.next_to_color = next_to_color
        self.relative_direction = relative_direction

    def end_neighbor(self):
        return self.game.get_piece(self.next_to_color, self.next_to_id)

    def end_position(self):
        return target_position(self.end_neighbor().position, self.relative_direction)

    def can_be_played(self):
        if self.end_neighbor() is None:
            return False
        return super().can_be_played()

    def __repr__(self):
        return '[Ant{} -> [{}, {}]]'.format(
            self.piece_id, self.end_neighbor(), self.relative_direction)


def create_all_actions(game, color):
    available_actions = []
    # Initial Deployment
    for piece_type in GamePieceType.PIECE_COUNT.keys():
        available_actions.append(InitialDeployAction(game, color, piece_type))
    # All Deployments
    for piece_type in GamePieceType.PIECE_COUNT.keys():
        for i in range(sum(GamePieceType.PIECE_COUNT.values())):
            for relative_direction in HiveGame.NEIGHBORS_DIRECTION:
                deploy = DeployAction(game, color, piece_type, i, relative_direction)
                available_actions.append(deploy)
    # Grasshopper
    for index in GamePieceType.PIECE_INDEX[GamePieceType.GRASSHOPPER]:
        for relative_direction in HiveGame.NEIGHBORS_DIRECTION:
            move = GrasshopperMove(game, color, index, relative_direction)
            available_actions.append(move)
    # Beetle
    for index in GamePieceType.PIECE_INDEX[GamePieceType.BEETLE]:
        for relative_direction in HiveGame.NEIGHBORS_DIRECTION:
            move = BeetleMove(game, color, index, relative_direction)
            available_actions.append(move)
    # Queen
    for index in GamePieceType.PIECE_INDEX[GamePieceType.QUEEN_BEE]:
        for relative_direction in HiveGame.NEIGHBORS_DIRECTION:
            move = QueenMove(game, color, index, relative_direction)
            available_actions.append(move)
    # Spider
    for index in GamePieceType.PIECE_INDEX[GamePieceType.SPIDER]:
        for x, y in product(range(-3, 4), range(-3, 4)):
            if hex_distance((0, 0), (x, y)) <= 3:
                move = SpiderMove(game, color, index, (x, y))
                available_actions.append(move)
    # Ant
    for index in GamePieceType.PIECE_INDEX[GamePieceType.SOLDIER_ANT]:
        for neighbor_index in range(sum(GamePieceType.PIECE_COUNT.values())):
            for neighbor_color in range(2):
                if neighbor_index == index and color == neighbor_color:
                    continue
                for relative_direction in HiveGame.NEIGHBORS_DIRECTION:
                    deploy = AntMove(game, color, index, neighbor_index, neighbor_color, relative_direction)
                    available_actions.append(deploy)
    return available_actions
