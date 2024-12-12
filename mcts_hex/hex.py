import numpy as np
from copy import deepcopy


class GameWorld:
    """Providing all game logic related to Hex."""
    def __init__(self, board_size: int) -> None:
        print('Gameworld instance is created')
        self.size: int = board_size
        #self.gameover_lookup: dict = {} # For efficiency: save knowledge about terminal states and winner

    def get_initial_state(self) -> np.ndarray:
        """Create an inital state, according to board size."""
        state = np.zeros(self.size**2 + 1, dtype='uint8')
        state[-1] = 1 #player 1 starts
        return state

    def get_board_size(self) -> int:
        return self.size

    def get_successor_state(self, prev_state: np.ndarray, action: int) -> np.ndarray:
        """Return next state given previous state and an action."""
        state = deepcopy(prev_state)
        if state[-1] == 1: # If player 1
            state[action] = 1 # Make the move
            state[-1] = 2 # Switch player's turn
        elif state[-1] == 2:
            state[action] = 2
            state[-1] = 1
        return state

    def get_all_successor_states(self, state: np.ndarray) -> np.ndarray:
        """Return all possible successor states from a given state."""
        actions = self.get_legal_actions(state)
        states = np.zeros((len(actions), self.size**2+1), dtype='uint8')
        for i, action in enumerate(actions):
            new_state = self.get_successor_state(state, action)
            states[i,:] = new_state
        return states

    def get_legal_actions(self, state: np.ndarray) -> np.ndarray:
        """Find all empty cells on board, each cell is a possible action."""
        return np.where(state == 0)[0]

    def check_if_win(self, state: np.ndarray) -> tuple:
        """Check if gameover, and if so, return the winner."""

        #state_str = np.array_str(state)
        #if state_str in self.gameover_lookup:
        #    return self.gameover_lookup[state_str]

        is_terminal = False
        winner_id = None
        win_path = []
        if state[-1] == 1: # Check if player 2 won.
            # Win is only possible with at least 'self.size' number of pieces present.
            if np.count_nonzero(state == 2) >= self.size: 
                for i in range(self.size):
                    if state[i*self.size] == 2: # If player 2 has an edge piece.
                        pos = (i,0)
                        is_connected, win_path = self._is_connected_to_edge(state, pos, player_id=2)
                        if is_connected:
                            is_terminal = True
                            winner_id = 2
        elif state[-1] == 2: # Check if player 1 won.
            # Win is only possible with at least 'self.size' number of pieces present.
            if np.count_nonzero(state == 1) >= self.size:
                for i in range(self.size):
                    if state[i] == 1: # If player 1 has an edge piece.
                        pos = (0,i)
                        is_connected, win_path = self._is_connected_to_edge(state, pos, player_id=1)
                        if is_connected:
                            is_terminal = True
                            winner_id = 1

        #self.gameover_lookup[state_str] = (is_terminal, winner_id)
        return is_terminal, winner_id, win_path

    def _get_neighbours(self, pos: tuple) -> list:
        """Return list of neighbours for a given board coordinate."""
        r = pos[0]
        c = pos[1]
        neighbours = []
        if r > 0:
            neighbours.append((r - 1, c))
            if c + 1 < self.size:
                neighbours.append((r - 1, c + 1))
        if r + 1 < self.size:
            neighbours.append((r + 1, c))
            if c > 0:
                neighbours.append((r + 1, c - 1))
        if c > 0:
            neighbours.append((r, c - 1))
        if c + 1 < self.size:
            neighbours.append((r, c + 1))
        return neighbours

    def _is_connected_to_edge(self, state: np.ndarray, pos: tuple, player_id: int) -> bool:
        """Return 1 if 'player' has a path from 'pos' to the other side."""
        visited = []
        path = [pos]
        def is_connected(pos):
            """Recursice call to determine if there is a path"""
            visited.append(pos)
            if player_id==1 and pos[0] == self.size-1:
                return True
            if player_id==2 and pos[1] == self.size-1: 
                return True
            neighbours = self._get_neighbours(pos)
            for neighbour in neighbours:
                if state[self.size*neighbour[0]+neighbour[1]] == player_id and neighbour not in visited:
                    connected = is_connected(neighbour)
                    if connected:
                        path.append(neighbour)
                        return True
            return False # no more unvisited neighbours 
        
        connected = is_connected(pos)
        if connected:
            return connected, path
        else:
            return connected, []

    def get_reward(self, winner_id: int) -> int:
        """Return award for winning, given who the player is."""
        if winner_id == 1:
            return 1
        elif winner_id == 2:
            return -1
