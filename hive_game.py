import numpy as np
from collections import defaultdict


class GamePieceType:
    QUEEN_BEE = 'queen_bee'
    BEETLE = 'beetle'
    GRASSHOPPER = 'grasshopper'
    SPIDER = 'spider'
    SOLDIER_ANT = 'soldier_ant'

    PIECE_COUNT = {
        QUEEN_BEE: 1,
        BEETLE: 2,
        GRASSHOPPER: 3,
        SPIDER: 2,
        SOLDIER_ANT: 3
    }


class GamePiece:
    def __init__(self, hive_game, color, piece_type, position):
        self.game = hive_game
        self.color = color
        self.piece_type = piece_type
        self.position = position

    def __repr__(self):
        return '{}[{}]({}, {}, {})'.format(
            self.piece_type, self.color, self.position[0], self.position[1], self.position[2])


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

        # First two positions are (0, 0) and (0, 1)
        if len(player_pieces) == 0:
            if self.position[0] != 0 or self.position[1] != len(enemy_pieces):
                return False
        return True

    def activate(self):
        new_piece = GamePiece(
            self.game, self.color, self.piece_type, (self.position[0], self.position[1], 0))
        self.game.set_top_piece(self.position, new_piece)
        if self.piece_type == GamePieceType.QUEEN_BEE:
            self.game.queens[self.color] = new_piece


class MoveAction(Action):
    def __init__(self, hive_game, start_position, end_position):
        super().__init__(hive_game)
        self.start_position = start_position
        self.end_position = end_position

    def _set_dfs_state(self, speed):
        all_positions = np.array([x.position for x in self.game.all_pieces()])
        self.min_x, self.max_x = np.min(all_positions[:, 0]) - 1, np.max(all_positions[:, 0]) + 1
        self.min_y, self.max_y = np.min(all_positions[:, 1]) - 1, np.max(all_positions[:, 1]) + 1
        self.dfs_found = False
        self.speed = speed

    def _dfs(self, current, level, used):
        #print(used)
        if current == self.end_position:
            if self.speed == np.inf or self.speed == level:
                self.dfs_found = True
                return
        if level >= self.speed:
            return
        for dx, dy in HiveGame.NEIGHBORS_DIRECTION:
            new_x, new_y = current[0] + dx, current[1] + dy
            if new_x < self.min_x or new_x > self.max_x or new_y < self.min_y or new_y > self.max_y:
                continue
            if self.game.get_top_piece((new_x, new_y)) is not None:
                continue
            if len(self.game.neighbor_pieces((new_x, new_y))) == 0:
                continue
            if (new_x, new_y) in used:
                continue
            # add tiny hole check here
            ##########################
            new_used = used.copy()
            new_used.update({(new_x, new_y)})
            #print(new_x, new_y)
            self._dfs((new_x, new_y), level + 1, new_used)
            if self.dfs_found:
                return

    def _slide_movement_allowed(self, speed):
        self._set_dfs_state(speed)
        self._dfs(self.start_position, 0, {self.start_position})
        return self.dfs_found

    def can_be_played(self):
        moving_piece = self.game.get_top_piece(self.start_position)
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
            return self._slide_movement_allowed(np.inf)
        elif moving_piece.piece_type == GamePieceType.SPIDER:
            return self._slide_movement_allowed(3)
        elif moving_piece.piece_type == GamePieceType.QUEEN_BEE:
            return self._slide_movement_allowed(1)
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
            if x_step == 1 or y_step == 1:
                return False

            # all pieces jumped over are filled
            for step in range(1, np.max(np.abs([dx, dy]))):
                if self.game.get_top_piece((self.start_position[0] + step * x_step,
                                            self.start_position[1] + step * y_step)) is None:
                    return False
            return True
        else:
            raise ValueError('Invalid game piece')

    def activate(self):
        pass


class HiveGame:
    NEIGHBORS_DIRECTION = [[0, -1], [0, 1], [-1, 0], [1, 0], [1, -1], [-1, 1]]
    BOARD_SIZE = 2 * np.sum(list(GamePieceType.PIECE_COUNT.values()))
    MAX_STACK_SIZE = 5

    def __init__(self):
        self.to_play = np.random.choice([0, 1], 1)[0]
        self._pieces = defaultdict(list)
        self.queens = [None, None]

    def get_winner(self):
        if None in self.queens:
            return None
        for queen in self.queens:
            if len(self.neighbor_pieces(queen.position)) == 6:
                return (queen.color + 1) % 2
        return None

    def play_action(self, action):
        assert action.can_be_played()
        action.activate()
        self.to_play = (self.to_play + 1) % 2

    def get_top_piece(self, position):
        if len(self._pieces[position]) == 0:
            return None
        return self._pieces[position][-1]

    def set_top_piece(self, position, piece):
        self._pieces[position].append(piece)

    def drop_top_piece(self, position):
        self._pieces[position] = self._pieces[position][:-1]

    def all_pieces(self):
        return [x for stack in self._pieces.values() for x in stack]

    def player_pieces(self, color):
        return [x for x in self.all_pieces() if x.color == color]

    def neighbor_pieces(self, position):
        pieces = []
        for dx, dy in HiveGame.NEIGHBORS_DIRECTION:
            piece = self.get_top_piece((position[0] + dx, position[1] + dy))
            if piece is not None:
                pieces.append(piece)
        return pieces

    def is_connecting_piece(self, piece):
        neighbor_pieces = self.neighbor_pieces(piece.position)
        if len(neighbor_pieces) <= 1:
            return False
        used = {piece.position}
        queue = [neighbor_pieces[0]]
        while len(queue) > 0:
            current, *queue = queue
            if current.position in used:
                continue
            used.add(current.position)
            neighbor_pieces = self.neighbor_pieces(current.position)
            queue.extend([x for x in neighbor_pieces if x.position not in used])
        return len(used) != len(self.all_pieces())


