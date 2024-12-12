from visualize_hex import HexVisualizer
from hex import GameWorld
import tensorflow as tf
import numpy as np
import random
from enum import Enum, auto


class InteractiveGame:
    """
        Run an interactive game between a previously trained model
        and a human player, providing input by clicking on a GUI.    
    """
    def __init__(self, board_size) -> None:
        self.size = board_size
        filename = f'models/set2/hex_size{board_size}_episode400'
        self.model = self.load_model(filename)
        self.game = GameWorld(board_size=board_size)
        self.viz = HexVisualizer(size=board_size)

    def load_model(self, filename):
        """Load model, prevously trained and saved."""
        model = tf.keras.models.load_model(filename)
        return model

    def get_anet_action(self, state: np.ndarray):
        """Interpret probability distribution from model as an action."""
        predictions: tf.Tensor = self.model(np.array([state]))[0]
        predictions = predictions.numpy()
        mask = np.zeros(predictions.shape[0])
        legal_actions = self.game.get_legal_actions(state)
        mask[legal_actions] = 1
        predictions = np.multiply(predictions, mask)
        action = np.argmax(predictions)
        if action not in legal_actions: # If distribution is zero for all legal moves
            action = legal_actions[random.randint(0,len(legal_actions)-1)]
        return action

    def match(self):
        """Run match between loaded model and human player, with realtime GUI."""
        state = self.game.get_initial_state()
        gameover = False
        current_player = Players.AI_AGENT # equals player 1
        self.viz.show(state)
        while not gameover:
            if current_player == Players.AI_AGENT:
                action = self.get_anet_action(state)
                current_player = Players.HUMAN
            elif current_player == Players.AI_AGENT:
                self.viz.wait_for_click()
                action = self.viz.detect_clicked_node()
                current_player = Players.HUMAN
            state = self.game.get_successor_state(state, action)
            gameover, winner, path = self.game.check_if_win(state)
            self.viz.show(state, path, winner)
            self.viz.pause_animation()
        self.viz.last_show()


class Players(Enum):
    HUMAN = auto()
    AI_AGENT = auto()


if __name__ == '__main__':
    print('\n\n')
    ai_game = InteractiveGame(6)
    ai_game.match()
    