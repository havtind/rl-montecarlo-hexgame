import tensorflow as tf
from tensorflow import keras
import numpy as np
from mcts_hex.mcts import GameWorld
import random


class TOPP:
    def __init__(self, board_size: int, episodes: list, relative_path: str=None) -> None:
        self.num_players = len(episodes)
        self.episodes: list = episodes
        self.players_id = np.arange(0, self.num_players)
        self.loaded_models: list = self.load_models(board_size)
        self.game = GameWorld(board_size=board_size)
        self.games = 25
        self.win_count = np.zeros(self.num_players)


    def load_models(self, board_size):
        directory = ''
        loaded_models = []
        for episode in self.episodes:
            file_name = f'hex_size{board_size}_episode{episode}'
            model = tf.keras.models.load_model(directory+file_name)
            loaded_models.append(model)
        return loaded_models

    def generate_series(self):
        series = []
        for i in range(self.num_players):
            for j in range(i+1, self.num_players):
                serie = (i, j)
                series.append(serie)
        return series

    def round_robin(self):
        print('\n \n Round robin started!')
        for serie in self.generate_series():
            player1, player2 = serie
            print(f'    Playing series between {player1} and {player2}:')
            for game in range(self.games):
                print(f'       Match {game+1}/{self.games}\r', end='')
                self.match(player1, player2)
                temp = player1
                player1 = player2
                player2 = temp
        print(f'\n \n Win stats: {self.win_count} ')
        

    def match(self, player_id_1, player_id_2):
        state = self.game.get_initial_state()
        gameover = False
        current_player = player_id_1
        while not gameover:
            action = self.get_action(state, current_player)
            state = self.game.get_successor_state(state, action)
            if current_player == player_id_1:
                current_player = player_id_2
            elif current_player == player_id_2:
                current_player = player_id_1
            gameover, winner, _ = self.game.check_if_win(state)
        if winner == 1:
            self.win_count[player_id_1] += 1
        elif winner == 2:
            self.win_count[player_id_2] += 1
        

    def get_action(self, state: np.ndarray, player_id):
        predictions: tf.Tensor = self.loaded_models[player_id](np.array([state]))[0]
        predictions = predictions.numpy()
        mask = np.zeros(predictions.shape[0])
        legal_actions = self.game.get_legal_actions(state)
        mask[legal_actions] = 1
        predictions = np.multiply(predictions, mask)
        action = np.argmax(predictions)
        if action not in legal_actions: # If distribution is zero for all legal moves
            action = legal_actions[random.randint(0,len(legal_actions)-1)]
        return action



def test():
    topp = TOPP(board_size=5, episodes=[0, 50, 100, 150, 200])
    topp.round_robin()
   

if __name__ == '__main__':
    test()

