import matplotlib.pyplot as plt
import networkx as nx
import numpy as np


class HexVisualizer:
    def __init__(self, size):
        self.size = size
        self.grid = self._create_grid(size) # List array of cells 
        self._create_neighbourhood() # Add neighbours to each cell

        self.edges = self.get_edges()
        self.grid_graph = nx.Graph()
        self.grid_graph.add_edges_from(self.edges, color='blue')
        self.positions = self.get_cell_pos(self.grid_graph)
        self.clicked_coordinates = (0,0)

        self.figure = plt.figure()
        cid = self.figure.canvas.mpl_connect('button_press_event', self.onclick)
        plt.axis('off')

    def onclick(self, event):
        self.clicked_coordinates = (event.xdata, event.ydata)

    def detect_clicked_node(self):
        x, y = self.clicked_coordinates
        margin = 0.7
        for cell, pos in self.positions.items():
            if abs(pos[0]-x) < margin and abs(pos[1]-y) < margin:
                r = cell.pos[0]
                c = cell.pos[1]
                action = r*self.size + c
                return action
        return None
    
    def show(self, state: np.ndarray, win_path=[], winner=None):
        plt.clf
        if winner == 1:
            plt.title('You won, AI agent!')
        elif winner == 2:
            plt.title('You won, human!')
        elif state[-1] == 1:
            plt.title('It is your turn, AI agent..')
        elif state[-1] ==2:
            plt.title('It is your turn, human..')

        self.update_cells(state, win_path) 
        nx.draw_networkx_nodes(self.grid_graph, pos=self.positions, node_color=self.get_cell_colors())
        edge_colors, edge_widths = self.get_edge_attributes()
        nx.draw_networkx_edges(self.grid_graph, pos=self.positions, edge_color=edge_colors, width=edge_widths)

    def pause_animation(self):
        plt.pause(1)
        
    def wait_for_click(self):
        plt.waitforbuttonpress()

    def last_show(self):
        plt.show()

    def update_cells(self, state, win_path=[]):
        for r in range(self.size):
            for c in range(self.size):
                self.grid[r][c].value = state[r*self.size+c]
                if (r,c) in win_path:
                    self.grid[r][c].in_path = True


    def get_edges(self):
        edges = []
        for row in self.grid:
            for cell in row:
                for neighbor in cell.neighbours:
                    if neighbor is not None:
                        edges.append((cell, neighbor))
        return edges

    def get_edge_attributes(self):
        colors = []
        widths = []
        for edge in self.grid_graph.edges:
            if edge[0].in_path and edge[1].in_path:
                if edge[0].value == 1:
                    colors.append('#EC0C0C')
                elif edge[0].value == 2:
                    colors.append('#0C53D1')
                widths.append(3)
            elif edge[0].is_red_edge and edge[1].is_red_edge:
                colors.append('#EC0C0C')
                widths.append(6)
            elif edge[0].is_blue_edge and edge[1].is_blue_edge:
                colors.append('#0C53D1')
                widths.append(6)
            else:
                colors.append('#8E8E8E')
                widths.append(1)
        return colors, widths

    def get_cell_pos(self, grid_graph):
        positions = {}
        width = 10
        for cell in grid_graph:
            rpos = cell.pos[0]
            cpos = cell.pos[1]
            xpos = width + (-width/4) * rpos + width/4 * cpos
            ypos = width + (-width/4) * rpos + (-width/4) * cpos
            positions[cell] = (xpos, ypos)
        return positions

    def get_cell_colors(self):
        colors = []
        for cell in self.grid_graph:
            if cell.value == 1:
                if cell.in_path:
                    colors.append('#EC0C0C')
                else:
                    colors.append('#FF9292')
            elif cell.value == 2:
                if cell.in_path:
                    colors.append('#0C53D1')
                else:
                    colors.append('#79A4EF')
            else:
                colors.append('#DBDBDB')
        return colors

    def _create_grid(self, size):
        grid = []
        for r in range(size):
            row = []
            for c in range(size):
                node = Node(r,c)
                if r == 0 or r == size-1:
                    node.is_red_edge = True
                if c == 0 or c == size-1:
                    node.is_blue_edge = True
                row.append(node)
            grid.append(row)
        return grid

    def _create_neighbourhood(self):
        size = len(self.grid)
        for r in range(size):
            for c in range(size):
                neighbours = [(r - 1, c + 1), (r + 1, c - 1), (r + 1, c), (r, c - 1), (r, c + 1), (r - 1, c)]
                for nb in neighbours:
                    if self._valid_neighbour(nb, size):
                        node: Node = self.grid[r][c]
                        neighbour: Node = self.grid[nb[0]][nb[1]]
                        node.neighbours.append(neighbour)

    def _valid_neighbour(self, nb, size):
            if not(nb[0] < 0 or nb[0] >= size):
                if not(nb[1] < 0 or nb[1] >= size):
                    return True
            else:
                return False


class Node:
    def __init__(self, r, c):
        self.pos = (r, c)
        self.value = 0
        self.neighbours = []
        self.in_path = False
        self.is_red_edge = False
        self.is_blue_edge = False


def test():
    size = 3
    hex = HexVisualizer(size)
    state = np.array([0,2,0,1,0,0,0,0,0,2])
    hex.show(state)
    while True:
        hex.wait_for_click()
        action = hex.detect_clicked_node()
    hex.last_show()


if __name__ == '__main__':
    print('\n\n')
    test()
   
  
   