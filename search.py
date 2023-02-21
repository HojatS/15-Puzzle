import copy
import numpy as np
import heapq

# ENUMS
UP = 0
LEFT = 1
DOWN = 2
RIGHT = 3

goal = [[1,2,3],[4,5,6],[7,8,0]]
goal_flat = [1,2,3,4,5,6,7,8,0]
    
class Puzzle:
    def __init__(self, puzzle): 
        self.puzzle = puzzle
        self.parent = None
        self.enum = 0 
        self.gdistance = 0
        self.hdistance = self.heuristic()
        
        
    def child_states(self):
        
        #find neighbors of 0 and store in order: UP DOWN LEFT RIGHT
        neighbors = []
        i_zero , j_zero = self.find_zero()
        
        if i_zero+1 in range(3): #UP
            neighbors.append([i_zero+1,j_zero, 0])
        if i_zero-1 in range(3): #DOWN
            neighbors.append([i_zero-1,j_zero, 2])
        if j_zero+1 in range(3): #left
            neighbors.append([i_zero,j_zero+1, 1])
        if j_zero-1 in range(3): #right
            neighbors.append([i_zero,j_zero-1, 3])
        
        puzzles = []
        #swap neighbors with 0 to produce new child states
        for neighbor in neighbors:
            child_state = Puzzle(self.swap(neighbor[0], neighbor[1]))
            child_state.enum = neighbor[2]
            child_state.parent = self
            child_state.gdistance = self.gdistance + 1
            if(self.parent == None):
                puzzles.append(child_state)
            elif(child_state.puzzle != self.parent.puzzle):
                puzzles.append(child_state)
                
        return puzzles     

    #find indeces of the location of 0
    def find_zero(self):
        temp = np.array(self.puzzle, dtype=int)
        location =  np.where(temp==0)
        i,j = location[0][0], location[1][0]
        return i,j

    #swaps 0 with a selected index
    def swap(self, index1, index2):
        child = copy.deepcopy(self.puzzle)
        i,j = self.find_zero()
        child[i][j], child[index1][index2] = child[index1][index2], child[i][j]
        return child

    def TotalDistance(self):
        return self.gdistance + self.hdistance
    
    def __gt__(self, other):
        if (self.TotalDistance() == other.TotalDistance()):
            return self.enum > other.enum
        else:
            return self.TotalDistance() > other.TotalDistance()

    def __lt__(self, other):
        if (self.TotalDistance() == other.TotalDistance()):
            return self.enum < other.enum
        else:
            return self.TotalDistance() < other.TotalDistance()

    
    #
    # A-Star with the heuristic: # of misplaced tiles 
    #  
    def heuristic(self):
        distance = 0
        temp = copy.deepcopy(self.puzzle)
        temp = np.array(temp)
        temp = temp.flatten()
        arr = list(temp)
        enum = enumerate(arr)
        for index, tile in enum:
            if tile != goal_flat[index]:
                distance = distance+1
        return distance

    """""
    #
    # A-Star with the heuristic: Sum of Manhattan Distance of each tile from correct position
    #
    def heuristic(self):
        distance = 0
        temp = copy.deepcopy(self.puzzle)
        temp = np.array(temp)
        temp = temp.flatten()
        arr = list(temp)
        enum = enumerate(arr)
        for index, tile in enum:
            if tile != goal_flat[index]:
                distance = distance + Manhattan_distance(index,goal_flat.index(tile))
        return distance

    """""
def Manhattan_distance(i, j):
    distance = 0
    X1= i // 3
    Y1= i % 3
    X2= j // 3
    Y2= j % 3
    return abs(X2 - X1) + abs(Y2 - Y1)


def toInt(array):
    for i in range(len(array)):
        for j in range(len(array)):
            array[i][j] = int(array[i][j])
    return array

    

def BFS(board):
    final_solution = []
    puzzle=Puzzle(toInt(board))
    visited = []
    queue = [puzzle]
    while queue:
        state = queue.pop(0)

        if state.puzzle == goal:
            visited.append(state)
            break
        if state not in visited:
            visited.append(state)
            for child in state.child_states():
                queue.append(child)

    for node in visited:
        final_solution.append(node.enum)
    
    del final_solution[0]
    return final_solution


def DFS(board):
    
    final_solution = []

    puzzle=Puzzle(toInt(board))

    visited = []
    stack = [puzzle]
    while stack:
        state = stack.pop()
        
        if state.puzzle == goal:
            visited.append(state)
            break
        if state not in visited:
            No_of_children = len(state.child_states())
            children = state.child_states()
            visited.append(state)
            for i in range(No_of_children):
                stack.append(children[No_of_children - i - 1])
        

    for node in visited:
        final_solution.append(node.enum)

    del final_solution[0]
    return final_solution


def A_Star_H1(board):

    puzzle=Puzzle(toInt(board))

    final_solution = []
    visited = []
    queue = []

    
    heapq.heappush(queue, puzzle)
    while queue:
        state = heapq.heappop(queue)
        
        if state.puzzle == goal:
            visited.append(state)
            break 
        if state not in visited:
            visited.append(state)
            for child in state.child_states():
                heapq.heappush(queue, child)

    for node in visited:
        final_solution.append(node.enum)

    del final_solution[0]
    return final_solution


def A_Star_H2(board):

    puzzle=Puzzle(toInt(board))

    final_solution = []
    visited = []
    queue = []

    
    heapq.heappush(queue, puzzle)
    while queue:
        state = heapq.heappop(queue)
        
        if state.puzzle == goal:
            visited.append(state)
            break 
        if state not in visited:
            visited.append(state)
            for child in state.child_states():
                heapq.heappush(queue, child)

    for node in visited:
        final_solution.append(node.enum)

    del final_solution[0]
    return final_solution






"""""
print("BFS")
print(BFS(puzzle))
print(get_move_string(BFS(puzzle)))

print("\nDFS")
print(DFS(puzzle))
print(get_move_string(DFS(puzzle)))
print("\nA_Star_H1")
print(A_Star_H1(puzzle))
print(get_move_string(A_Star_H1(puzzle)))
print("\nA_Star_H2")
print(A_Star_H2(puzzle))
print(get_move_string(A_Star_H2(puzzle)))
"""""
