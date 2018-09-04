import numpy as np
from itertools import product

from engine.hive_game import HiveGame
from engine.game_piece import GamePieceType


def target_position(start_position, relative_direction):
    return (start_position[0] + relative_direction[0],
            start_position[1] + relative_direction[1])


def hex_distance(a, b):
    return (abs(a[0] - b[0]) + abs(a[0] + a[1] - b[0] - b[1]) + abs(a[1] - b[1])) / 2


class Action:
    def __init__(self):
        self.game = None
        self.debug = False

    def _is_available(self, common_data):
        raise NotImplemented()

    def uniform_representation(self):
        raise NotImplemented()

    def _init(self):
        pass

    def can_be_played(self, hive_game, common_data=None):
        self.game = hive_game
        self._init()
        return self._is_available(common_data)


class InitialDeployAction(Action):
    def __init__(self, piece_type):
        super().__init__()
        self.piece_type = piece_type

    def _can_be_played(self):
        # Wrong turn
        if self.game.to_play != self.game.to_play:
            return False

        player_pieces = self.game.player_pieces(self.game.to_play)
        if len(player_pieces) != 0:
            return False

        return True

    def _is_available(self, common_data=None):
        common_data_key = 'initial', self.game.to_play
        result = (common_data or {}).get(common_data_key)
        if result is not None:
            return result
        result = self._can_be_played()
        if common_data is not None:
            common_data[common_data_key] = result
        return result

    def _position(self):
        enemy_pieces = self.game.player_pieces((self.game.to_play + 1) % 2)
        return (0, 0) if len(enemy_pieces) == 0 else (1, 0)

    def activate(self):
        position = self._position()
        self.game.deploy_piece(position, self.piece_type)

    def uniform_representation(self):
        return 'initial', self.piece_type, self._position()

    def __repr__(self):
        return 'Initial {}'.format(self.piece_type)


class DeployAction(Action):
    def __init__(self, piece_type, next_to_id, relative_direction):
        super().__init__()
        self.piece_type = piece_type
        self.next_to_id = next_to_id
        self.relative_direction = relative_direction

    def _get_real_position(self):
        neighbor = self.game.get_piece(self.game.to_play, self.next_to_id)
        return target_position(neighbor.position, self.relative_direction)

    def _can_deploy_there(self):
        # Wrong turn
        if self.game.to_play != self.game.to_play:
            return False

        neighbor = self.game.get_piece(self.game.to_play, self.next_to_id)
        if neighbor is None:
            return False

        player_pieces = self.game.player_pieces(self.game.to_play)

        # No queen on 4th turn
        if (self.game.get_piece(self.game.to_play, 0) is None and self.piece_type != GamePieceType.QUEEN_BEE
                and len(player_pieces) == 3):
            return False

        # Position is empty
        position = self._get_real_position()
        if self.game.get_top_piece(position) is not None:
            return False
        neighbor_pieces_colors = {x.color for x in self.game.neighbor_pieces(position)}

        # Doesn't touch enemies
        if (self.game.to_play + 1) % 2 in neighbor_pieces_colors:
            return False

        return True

    def _is_available(self, common_data=None):
        common_data_key = 'deploy', self.game.to_play, self.next_to_id, self.relative_direction
        result = (common_data or {}).get(common_data_key)
        if result is None or self.piece_type != GamePieceType.QUEEN_BEE:
            result = self._can_deploy_there()
            if common_data is not None:
                common_data[common_data_key] = result
        if not result:
            return result

        player_pieces = self.game.player_pieces(self.game.to_play)
        has_pieces_left = len([x for x in player_pieces
                               if x.piece_type == self.piece_type]) < GamePieceType.PIECE_COUNT[self.piece_type]
        return has_pieces_left

    def activate(self):
        position = self._get_real_position()
        self.game.deploy_piece(position, self.piece_type)

    def uniform_representation(self):
        return 'deploy', self.piece_type, self._get_real_position()

    def __repr__(self):
        position = self._get_real_position()
        return '{}[{}, {}]'.format(self.piece_type, position[0], position[1])


class BaseMove(Action):
    def __init__(self, piece_id):
        super().__init__()
        self.piece_id = piece_id

    def end_position(self):
        raise NotImplemented()

    def _moving_piece(self):
        return self.game.get_piece(self.game.to_play, self.piece_id)

    def _is_available(self, common_data=None):
        common_data_key = 'base_move', self.game.to_play, self.piece_id
        result = (common_data or {}).get(common_data_key)
        if result is not None:
            return result
        result = self._can_be_played()
        if common_data is not None:
            common_data[common_data_key] = result
        return result

    def _init(self):
        self.moving_piece = self._moving_piece()

    def _can_be_played(self):
        # Wrong turn
        if self.game.to_play != self.game.to_play:
            return False

        # Piece is deployed
        if self.moving_piece is None:
            return False

        # Piece is on top
        if not self.moving_piece.is_on_top():
            return False

        # Moving player has queen deployed
        if self.game.get_piece(self.game.to_play, 0) is None:
            return False

        # The piece is non-connecting
        if self.game.is_connecting_piece(self.moving_piece):
            return False

        return True

    def activate(self):
        self.moving_piece = self._moving_piece()
        self.game.drop_top_piece(self.moving_piece.position)
        end_position = self.end_position()
        self.game.set_top_piece(end_position, self.moving_piece)
        self.moving_piece.position = end_position

    def uniform_representation(self):
        return 'move', self.moving_piece.position, self.end_position()


class GrasshopperMove(BaseMove):
    def __init__(self, piece_id, relative_direction):
        super().__init__(piece_id)
        self.relative_direction = relative_direction

    def _simulate_jump(self):
        jumped_over = []
        current_position = self.moving_piece.position
        while True:
            current_position = target_position(current_position, self.relative_direction)
            jumped_over_piece = self.game.get_top_piece(current_position)
            if jumped_over_piece is None:
                break
            jumped_over.append(jumped_over_piece)
        return current_position, jumped_over

    def _is_available(self, common_data=None):
        if not super()._is_available(common_data):
            return False

        jump_position, jumped_over = self._simulate_jump()
        if len(jumped_over) == 0:
            return False
        return True

    def end_position(self):
        return self._simulate_jump()[0]

    def __repr__(self):
        return '[Grasshopper{} -> [{}, {}]]'.format(
            self.piece_id, self.relative_direction[0], self.relative_direction[1])


class BeetleMove(BaseMove):
    def __init__(self, piece_id, relative_direction):
        super().__init__(piece_id)
        self.relative_direction = relative_direction

    def _is_available(self, common_data=None):
        if not super()._is_available(common_data):
            return False

        new_position = target_position(self.moving_piece.position, self.relative_direction)
        target_neighbor_pieces = {x for x in self.game.neighbor_pieces(new_position) if x != self.moving_piece}
        neighbor_pieces = {x for x in self.game.neighbor_pieces(self.moving_piece.position)}
        # No running away from hive and slide on pieces
        if len(target_neighbor_pieces) == 0 or len(neighbor_pieces.intersection(target_neighbor_pieces)) == 0:
            return False

        # Freedom to move rule for beetles
        beetle_stack_size = len(self.game.get_stack(self.moving_piece.position)) - 1
        target_stack_size = len(self.game.get_stack(new_position))
        index_direction = HiveGame.NEIGHBORS_DIRECTION.index(self.relative_direction)
        prev_direction = HiveGame.NEIGHBORS_DIRECTION[(index_direction - 1) % len(HiveGame.NEIGHBORS_DIRECTION)]
        next_direction = HiveGame.NEIGHBORS_DIRECTION[(index_direction + 1) % len(HiveGame.NEIGHBORS_DIRECTION)]
        gate1_size = len(self.game.get_stack(target_position(self.moving_piece.position, prev_direction)))
        gate2_size = len(self.game.get_stack(target_position(self.moving_piece.position, next_direction)))
        if (beetle_stack_size < gate1_size and beetle_stack_size < gate2_size and
                target_stack_size < gate1_size and target_stack_size < gate2_size):
            return False
        return True

    def end_position(self):
        return target_position(self.moving_piece.position, self.relative_direction)

    def __repr__(self):
        return '[Beetle{} -> [{}, {}]]'.format(
            self.piece_id, self.relative_direction[0], self.relative_direction[1])


class QueenMove(BeetleMove):
    def __init__(self, piece_id, relative_direction):
        super().__init__(piece_id, relative_direction)

    def _is_available(self, common_data=None):
        if not super()._is_available(common_data):
            return False

        # Can't go on top like beetle
        new_position = target_position(self.moving_piece.position, self.relative_direction)
        if self.game.get_top_piece(new_position) is not None:
            return False

        return True

    def __repr__(self):
        return '[Queen{} -> [{}, {}]]'.format(
            self.piece_id, self.relative_direction[0], self.relative_direction[1])


class ComplexMove(BaseMove):
    def _calculate_reachable(self, moving_piece):
        raise NotImplemented()

    def _next_spaces(self, current, used):
        current_neighbors = self.game.neighbor_pieces(current)
        for index_direction, (dx, dy) in enumerate(HiveGame.NEIGHBORS_DIRECTION):
            new_x, new_y = current[0] + dx, current[1] + dy
            if self.game.get_top_piece((new_x, new_y)) is not None:
                continue
            neighbor_pieces = {x for x in self.game.neighbor_pieces((new_x, new_y)) if x != self.moving_piece}
            if len(neighbor_pieces) == 0 or len(current_neighbors.intersection(neighbor_pieces)) == 0:
                continue
            if (new_x, new_y) in used:
                continue

            prev_direction = HiveGame.NEIGHBORS_DIRECTION[(index_direction - 1) % len(HiveGame.NEIGHBORS_DIRECTION)]
            next_direction = HiveGame.NEIGHBORS_DIRECTION[(index_direction + 1) % len(HiveGame.NEIGHBORS_DIRECTION)]
            if (self.game.get_top_piece(target_position(current, prev_direction)) is not None and
                    self.game.get_top_piece(target_position(current, next_direction)) is not None):
                continue
            yield new_x, new_y

    def _is_available(self, common_data=None):
        if not super()._is_available(common_data):
            return False

        end_position = self.end_position()

        # Same start and end not allowed
        if self.moving_piece.position == end_position:
            return False

        # Position is not empty
        if self.game.get_top_piece(end_position) is not None:
            return False

        common_data_key = 'reachable_move', self.moving_piece.position, end_position
        all_reachable = (common_data or {}).get(common_data_key)
        if all_reachable is None:
            all_reachable = self._calculate_reachable(self.moving_piece)
            if common_data is not None:
                common_data[common_data_key] = all_reachable
        return end_position in all_reachable


class SpiderMove(ComplexMove):
    def __init__(self, piece_id, relative_direction):
        super().__init__(piece_id)
        self.relative_direction = relative_direction

    def end_position(self):
        return target_position(self.moving_piece.position, self.relative_direction)

    def _dfs(self, current, level, used, reachable):
        if level == 3:
            reachable.add(current)
        if level >= 3:
            return

        for new_x, new_y in self._next_spaces(current, used):
            new_used = used.copy()
            new_used.update({(new_x, new_y)})
            self._dfs((new_x, new_y), level + 1, new_used, reachable)

    def _calculate_reachable(self, moving_piece):
        reachable = set()
        self._dfs(moving_piece.position, 0, {moving_piece.position}, reachable)
        return reachable

    def __repr__(self):
        return '[Spider{} -> [{}, {}]]'.format(
            self.piece_id, self.relative_direction[0], self.relative_direction[1])


class AntMove(ComplexMove):
    def __init__(self, piece_id, next_to_id, next_to_color, relative_direction):
        super().__init__(piece_id)
        self.next_to_id = next_to_id
        self.next_to_color = next_to_color
        self.relative_direction = relative_direction

    def end_neighbor(self):
        return self.game.get_piece(self.next_to_color, self.next_to_id)

    def end_position(self):
        return target_position(self.end_neighbor().position, self.relative_direction)

    def _calculate_reachable(self, moving_piece):
        used = set()
        queue = [moving_piece.position]
        while len(queue) > 0:
            current, *queue = queue
            next_spaces = list(self._next_spaces(current, used))
            queue.extend(next_spaces)
            used.update(next_spaces)
        return used

    def _is_available(self, common_data=None):
        if self.end_neighbor() in [None, self.moving_piece]:
            return False
        return super()._is_available(common_data)

    def __repr__(self):
        return '[Ant{} -> [{}, {}]]'.format(
            self.piece_id, self.end_neighbor(), self.relative_direction)


def create_all_actions():
    all_actions = []
    # Initial Deployment
    for piece_type in GamePieceType.PIECE_COUNT.keys():
        all_actions.append(InitialDeployAction(piece_type))
    # All Deployments
    for piece_type in GamePieceType.PIECE_COUNT.keys():
        for i in range(sum(GamePieceType.PIECE_COUNT.values())):
            for relative_direction in HiveGame.NEIGHBORS_DIRECTION:
                deploy = DeployAction(piece_type, i, relative_direction)
                all_actions.append(deploy)
    # Grasshopper
    for index in GamePieceType.PIECE_INDEX[GamePieceType.GRASSHOPPER]:
        for relative_direction in HiveGame.NEIGHBORS_DIRECTION:
            move = GrasshopperMove(index, relative_direction)
            all_actions.append(move)
    # Beetle
    for index in GamePieceType.PIECE_INDEX[GamePieceType.BEETLE]:
        for relative_direction in HiveGame.NEIGHBORS_DIRECTION:
            move = BeetleMove(index, relative_direction)
            all_actions.append(move)
    # Queen
    for index in GamePieceType.PIECE_INDEX[GamePieceType.QUEEN_BEE]:
        for relative_direction in HiveGame.NEIGHBORS_DIRECTION:
            move = QueenMove(index, relative_direction)
            all_actions.append(move)
    # Spider
    for index in GamePieceType.PIECE_INDEX[GamePieceType.SPIDER]:
        for x, y in product(range(-3, 4), range(-3, 4)):
            if hex_distance((0, 0), (x, y)) <= 3:
                move = SpiderMove(index, (x, y))
                all_actions.append(move)
    # Ant
    for index in GamePieceType.PIECE_INDEX[GamePieceType.SOLDIER_ANT]:
        for neighbor_index in range(sum(GamePieceType.PIECE_COUNT.values())):
            for neighbor_color in range(2):
                for relative_direction in HiveGame.NEIGHBORS_DIRECTION:
                    deploy = AntMove(index, neighbor_index, neighbor_color, relative_direction)
                    all_actions.append(deploy)
    return all_actions
