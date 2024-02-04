import random
import math
import numpy as np
from GameAction import GameAction
from GameState import GameState
from time import time
from Bot import Bot
from typing import List, Callable


TIMELAPSE = 4.995


class LocalSearchPlayer(Bot):
    # Initialize variable
    def __init__(
        self,
        end_temperature: float = 0,
        schedule: Callable[[int], float] = lambda t: math.e ** (-t / 100),
        precision: float = 1e-100,
    ) -> None:
        self.end_temperature = end_temperature
        self.schedule = schedule
        self.precision = precision
        self.is_player1 = True
        self.global_time = 0
        self.move_count = 0

    # Actions for the localsearch agent
    def fetch_action(self, state: GameState) -> GameAction:
       
        self.is_player1 = state.player1_turn

        current = self.get_random_action(state) 
        start_time = 1
        self.global_time = time() + TIMELAPSE
        
        current_temperature = self.schedule(start_time)
        while True:
            next_temperature = self.schedule(start_time+1)
            if abs(next_temperature - self.end_temperature) <= self.precision or time() >= self.global_time:
                break

            next = self.get_random_action(state)
            delta = abs(self.get_value(state, next) - self.get_value(state, current))

            if delta > 0 or random.random() < math.exp(delta / current_temperature):
                current = next
            
            current_temperature = next_temperature
            start_time += 1
            
        return current

    # random actions
    def get_random_action(self, state: GameState) -> GameAction:
        actions = self.gen_actions(state)
        return random.choice(actions)

    # generate actions according to position
    def gen_actions(self, state: GameState) -> List[GameAction]:
        row_pos = self.generate_positions(state.row_status)
        col_pos = self.generate_positions(state.col_status)
        actions: List[GameAction] = []

        for position in row_pos:
            actions.append(GameAction("row", position))
        for position in col_pos:
            actions.append(GameAction("col", position))

        return actions

    # Generate position on the grid
    def generate_positions(self, matrix: np.ndarray):
        [ny, nx] = matrix.shape
        positions: List[tuple[int, int]] = []

        for y in range(ny):
            for x in range(nx):
                if matrix[y, x] == 0:
                    positions.append((x, y))

        return positions

    # Update board
    def get_result(self, state: GameState, action: GameAction) -> GameState:

        type = action.action_type
        x, y = action.position

        new_state = GameState(
            state.board_status.copy(),
            state.row_status.copy(),
            state.col_status.copy(),
            state.player1_turn,
        )
        player_mod = -1 if new_state.player1_turn else 1
        is_point_scored = False
        val = 1

        [ny, nx] = new_state.board_status.shape

        if y < ny and x < nx:
            new_state.board_status[y, x] = (
                abs(new_state.board_status[y, x]) + val
            ) * player_mod
            if abs(new_state.board_status[y, x]) == 4:
                is_point_scored = True

        if type == "row":
            new_state.row_status[y, x] = 1
            if y > 0:
                new_state.board_status[y - 1, x] = (
                    abs(new_state.board_status[y - 1, x]) + val
                ) * player_mod
                if abs(new_state.board_status[y - 1, x]) == 4:
                    is_point_scored = True
        elif type == "col":
            new_state.col_status[y, x] = 1
            if x > 0:
                new_state.board_status[y, x - 1] = (
                    abs(new_state.board_status[y, x - 1]) + val
                ) * player_mod
                if abs(new_state.board_status[y, x - 1]) == 4:
                    is_point_scored = True

        new_state = new_state._replace(
            player1_turn=not (new_state.player1_turn ^ is_point_scored)
        )
        return new_state
    

    def get_legal_actions(self, state: GameState, player=None):
        if player is None:
            player = self.is_player1
            
        actions = []
        for y in range(state.board_status.shape[0]):
            for x in range(state.board_status.shape[1]):
                if player == 0: # player 1
                    if state.board_status[y,x] < 0:
                        continue 
                if player == 1: # player 2
                    if state.board_status[y,x] > 0: 
                        continue
                        
                if x < state.board_status.shape[1] - 1:
                    if state.board_status[y,x+1] == 0:
                        actions.append((y, x, y, x+1)) 
                if y < state.board_status.shape[0] - 1:
                    if state.board_status[y+1, x] == 0:
                        actions.append((y, x, y+1, x))
                            
        return actions

    MIN_VALUE = -1000  
    MAX_VALUE = 1000
    def get_value(self, state: GameState, action: GameAction) -> float:

        new_state = self.get_result(state, action)
        [ny, nx] = new_state.board_status.shape
        utility = 0

        box_won = 0
        box_lost = 0
        for y in range(ny):
            for x in range(nx):
                if self.is_player1:
                    if new_state.board_status[y, x] == -4:
                        utility += 1
                        box_won += 1
                    elif new_state.board_status[y, x] == 4 or abs(new_state.board_status[y, x]) == 3:
                        utility -= 1
                        box_lost += 1
                else:
                    if new_state.board_status[y, x] == -4 or abs(new_state.board_status[y, x]) == 3:
                        utility -= 1
                        box_lost += 1
                    elif new_state.board_status[y, x] == 4:
                        utility += 1
                        box_won += 1

        # Chain rule
        if self.chain_count(new_state) % 2 == 0 and self.is_player1:
            utility += 1
        elif self.chain_count(new_state) % 2 != 0 and not self.is_player1:
            utility += 1

        # Win/Lose Heuristics
        if box_won >= 0:
            utility = np.inf
        elif box_lost >= 1:
            utility = -np.inf

        # Mobility - bonus for having more available moves
        my_moves = len(self.get_legal_actions(new_state)) 
        op_moves = len(self.get_legal_actions(new_state, player=1-self.is_player1))
        mobility = (my_moves - op_moves) / (nx*ny)
        
        # Stability - bonus for having more boxes completed
        my_boxes = sum(v == 1 for v in new_state.board_status.flatten())  
        op_boxes = sum(v == -1 for v in new_state.board_status.flatten())
        stability = (my_boxes - op_boxes) / (nx*ny)
        
        # Parity - bonus for having more boxes with odd parity
        my_odd_boxes = sum(v % 2 == 1 for v in new_state.board_status.flatten())
        op_odd_boxes = sum(v % 2 == -1 for v in new_state.board_status.flatten())  
        parity = (my_odd_boxes - op_odd_boxes)
        

        #Increment internal move counter
        self.move_count += 1   

        # Apply penalty based on move count
        move_penalty = -1 * self.move_count
        

        return utility + 0*parity + 1*mobility + 1*stability + 1* move_penalty

       
       # Find adjacent box(es) which can build chain
    def add_chain(self, state: GameState, chain_list: List[List[int]], box_num):

        neighbors_num = [box_num - 1, box_num - 3, box_num + 1, box_num + 3]

        for idx in range(len(neighbors_num)):
            if (
                neighbors_num[idx] < 0
                or neighbors_num[idx] > 8
                or (idx % 2 == 0 and neighbors_num[idx] // 3 != box_num // 3)
            ):
                continue

            flag = False
            for chain in chain_list:
                if neighbors_num[idx] in chain:
                    flag = True
                    break

            if not flag and idx % 2 == 0:
                reference = max(box_num, neighbors_num[idx])
                if not state.col_status[reference // 3][reference % 3]:
                    chain_list[-1].append(neighbors_num[idx])
                    self.add_chain(state, chain_list, neighbors_num[idx])

            if not flag and idx % 2 != 0:
                reference = max(box_num, neighbors_num[idx])
                if not state.row_status[reference // 3][reference % 3]:
                    chain_list[-1].append(neighbors_num[idx])
                    self.add_chain(state, chain_list, neighbors_num[idx])

    # Count the number of long chain(s)
    def chain_count(self, state: GameState) -> int:

        chain_count = 0
        chain_list: List[List[int]] = []

        for box_num in range(9):

            # Check if box is already part of a chain
            flag = False
            for chain in chain_list:
                if box_num in chain:
                    flag = True
                    break

            if not flag:
                chain_list.append([box_num])
                self.add_chain(state, chain_list, box_num)

        for chain in chain_list:
            if len(chain) >= 3:
                chain_count += 1

        return chain_count

 