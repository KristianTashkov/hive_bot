class GamePieceType:
    QUEEN_BEE = 'queen'
    BEETLE = 'beetle'
    SPIDER = 'spider'
    GRASSHOPPER = 'grasshopper'
    SOLDIER_ANT = 'ant'

    PIECE_INDEX = {
        QUEEN_BEE: [0],
        BEETLE: [1, 2],
        SPIDER: [3, 4],
        GRASSHOPPER: [5, 6, 7],
        SOLDIER_ANT: [8, 9, 10]
    }

    PIECE_COUNT = {key: len(value) for key, value in PIECE_INDEX.items()}


class GamePiece:
    def __init__(self, hive_game, color, piece_type, position, id):
        self.game = hive_game
        self.color = color
        self.piece_type = piece_type
        self.position = position
        self.id = id

    def full_position(self):
        return self.position[0], self.position[1], self.game.get_stack(self.position).index(self)

    def is_on_top(self):
        return self == self.game.get_top_piece(self.position)

    def __repr__(self):
        return '{}[{}]({}, {})'.format(
            self.piece_type, self.color, self.position[0], self.position[1])