import os
import cv2
import numpy as np
from itertools import product
import matplotlib.pyplot as plt
from matplotlib import animation
from IPython.display import HTML

from engine.game_piece import GamePieceType


def combine_images(base, top, top_mask, x=0, y=0):
    base = base.copy()
    top = top.copy()
    rows, cols, channels = top.shape
    prev_background = base[x:x + rows, y: y + cols]

    mask = np.full_like(top_mask, 0)
    mask[top_mask > 50] = 255
    mask_inv = cv2.bitwise_not(mask)
    img1_bg = cv2.bitwise_and(prev_background, prev_background, mask=mask_inv)
    img2_fg = cv2.bitwise_and(top, top, mask=mask)

    dst = cv2.add(img1_bg, img2_fg)
    base[x:x + rows, y: y + cols] = dst
    return base


class GameRenderer:
    def __init__(self, sprites_file=None):
        if sprites_file is None:
            sprites_file = os.path.join(os.path.dirname(os.path.realpath(__file__)), '../assets/tilesprites-small.png')
        sprites = cv2.imread(sprites_file, cv2.IMREAD_UNCHANGED)
        sprites = cv2.cvtColor(sprites, cv2.COLOR_RGBA2BGRA)
        sprite_height = 57
        sprite_width = 64

        piece_order = [GamePieceType.SOLDIER_ANT, GamePieceType.BEETLE, GamePieceType.GRASSHOPPER,
                       GamePieceType.QUEEN_BEE, GamePieceType.SPIDER]
        self.piece_images = {}
        self.piece_masks = {}
        self.tile_images = []

        for i, piece_type in enumerate(piece_order):
            self.piece_images[piece_type] = sprites[sprite_height * 3: sprite_height * 4,
                                                    sprite_width * i: (sprite_width * (i + 1))]
            self.piece_masks[piece_type] = sprites[sprite_height * 1: sprite_height * 2,
                                                   sprite_width * i: (sprite_width * (i + 1))]
            self.piece_masks[piece_type] = cv2.cvtColor(self.piece_masks[piece_type], cv2.COLOR_BGRA2GRAY)
        for tile_id in [0, 1, 4, 2]:

            self.tile_images.append(sprites[sprite_height * 6: sprite_height * 7,
                                            (sprite_width * tile_id):(sprite_width * (tile_id + 1))])
        self.tile_mask = cv2.cvtColor(self.tile_images[1], cv2.COLOR_BGRA2GRAY)

    def get_piece(self, piece_type, color, is_bottom):
        if is_bottom and color == 1:
            color = 3
        return combine_images(self.tile_images[color], self.piece_images[piece_type],
                              self.piece_masks[piece_type])

    def render(self, hive_game):
        board = np.full((1000, 1000, 4), 0, dtype=np.uint8)
        board[:, :, 1:3] = 255
        board[:, :, 3] = 255
        positions = [x.position for x in hive_game.all_pieces()]
        min_x, min_y = np.min([x[0] for x in positions]), np.min([x[1] for x in positions])
        max_x, max_y = np.max([x[0] for x in positions]), np.max([x[1] for x in positions])
        for x, y in product(range(max_x, min_x - 1, -1), range(min_y, max_y + 1)):
            stack = hive_game.get_stack((x, y))
            for i, piece in enumerate(stack):
                piece_image = self.get_piece(piece.piece_type, piece.color, i < len(stack) - 1)
                normalized_x = x - min(min_x, -9)
                normalized_y = y - min(min_y, -3)
                real_x = normalized_x * piece_image.shape[1]
                real_y = normalized_y * piece_image.shape[0]
                board = combine_images(board, piece_image, self.tile_mask,
                                       real_y + normalized_x * 32 - i * 5,
                                       real_x - normalized_x * 15 + i * 5)
        return board

    def game_review(self, hive_game):
        image_frames = [self.render(x) for x in hive_game.game_history]
        # First set up the figure, the axis, and the plot element we want to animate
        fig = plt.figure(figsize=(20, 20), dpi=50)
        game_render = plt.imshow(np.full((1000, 1000, 4), 255))

        def init():
            game_render.set_data(np.full((1000, 1000, 4), 255))
            return game_render,

        # animation function. This is called sequentially
        def animate(i):
            game_render.set_data(image_frames[i])
            return game_render,

        # call the animator. blit=True means only re-draw the parts that have changed.
        anim = animation.FuncAnimation(fig, animate, init_func=init,
                                       frames=len(image_frames), interval=500, blit=True)
        return HTML(anim.to_jshtml())