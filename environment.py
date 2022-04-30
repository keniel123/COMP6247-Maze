from enum import Enum, IntEnum
from utils import read_maze


class Action(IntEnum):
    MOVE_LEFT = 0
    MOVE_RIGHT = 1
    MOVE_UP = 2
    MOVE_DOWN = 3
    STAY = 4


class MOVEMENT(Enum):
    MOVE_LEFT = (-1, 0)
    MOVE_RIGHT = (1, 0)
    MOVE_UP = (0, 1)
    MOVE_DOWN = (0, -1)
    STAY = (0, 0)


class Environment:
    reach_goal = 10.0  # reward for reaching the exit cell
    penalty_move = -0.05  # penalty for a move which did not result in finding the exit cell
    penalty_visited = -0.25  # penalty for returning to a cell which was visited earlier
    penalty_impossible_move = -0.9  # penalty for trying to enter a cell with a wall
    penalty_fire = -2.0  # penalty for trying to enter a cell with fire
    penalty_cannot_move = -.6  # penalty for being in a state that the agent cannot move
    penalty_no_movement = -.3  # penalty for remaining in current state

    def __init__(self):
        super(Environment, self).__init__()
        self.total_reward = 0.0
        self.visited = set()
        self.current_state = []
        self.current_cell = (1, 1)
        self.agent_path = [self.current_cell]
        self.observation = self.get_observation
        self.minimum_reward = -100
        self.goal_cell = (199, 199)

    @property
    def reset(self):
        self.current_cell = (1, 1)
        self.total_reward = 0.0  # accumulated reward
        self.visited = set()  # a set() only stores unique values
        self.observation = self.get_observation
        return self.observation

    @property
    def get_observation(self):
        col, row = self.current_cell

        maze_state = read_maze.get_local_maze_information(col, row)
        self.current_state = maze_state.copy()
        self.observation = maze_state
        self.agent_path.append(self.current_cell)

        return self.observation.flatten()

    @property
    def get_current_cell(self):
        return self.current_cell

    @property
    def get_current_state(self):
        return self.current_state

    def step(self, action):
        col, row = self.get_current_cell
        impassible = True
        current_state = self.get_current_state

        if action == Action.MOVE_LEFT:
            movement = MOVEMENT.MOVE_LEFT
        elif action == Action.MOVE_UP:
            movement = MOVEMENT.MOVE_UP
        elif action == Action.MOVE_RIGHT:
            movement = MOVEMENT.MOVE_RIGHT
        elif action == Action.MOVE_DOWN:
            movement = MOVEMENT.MOVE_DOWN
        elif action == Action.STAY:
            movement = MOVEMENT.STAY

        for i in current_state:
            for x in i:
                if x[0] == 1 and x[1] == 0:  # no wall and no fire
                    impassible = False

        temp_move_col, temp_move_row = movement.value
        relative_col_location, relative_row_location = (1 + temp_move_col, 1 + temp_move_row)

        if action == Action.STAY:
            if impassible == True:
                return self.get_observation, self.penalty_no_movement, self.check_game_status
            else:
                return self.get_observation, self.penalty_cannot_move, self.check_game_status

        if current_state[relative_col_location][relative_row_location][0] == 0:
            return self.get_observation, self.penalty_impossible_move, self.check_game_status
        if current_state[relative_col_location][relative_row_location][1] > 0:
            return self.get_observation, self.penalty_fire, self.check_game_status

        self.current_cell = (col + relative_col_location, row + relative_row_location)
        self.visited.add(self.current_cell)
        if self.current_cell == self.goal_cell:
            return self.observation, self.goal_cell, self.check_game_status
        elif self.current_cell in self.visited:
            return self.get_observation, self.penalty_visited, self.check_game_status
        else:
            return self.get_observation, self.penalty_move, self.check_game_status

    def update_total_reward(self, reward):
        self.total_reward += reward

    @property
    def check_game_status(self):
        if self.current_cell == self.goal_cell or self.total_reward < self.minimum_reward:
            return True
        return False
