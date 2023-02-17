import numpy as np
import heapq

# ENUMS
UP = 0
LEFT = 1
DOWN = 2
RIGHT = 3

goal = [[1,2,3],[4,5,6],[7,8,0]]
    
class Puzzle:
    def __init__(self, puzzle): 
        self.puzzle = puzzle
        self.parent = None
        self.enum = None #ENUMS aka how we arrived from parent
        self.gdistance = 0
        self.hdistance = self.heuristic()

    def child_states(self):
        puzzles = []
        
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
        

        #swap neighbors with 0 to produce new child states
        for neighbor in neighbors:
            print(neighbor)
            #parent = self
            child_state = Puzzle(self.swap(neighbor[0], neighbor[1]))
            print(child_state.puzzle)
            child_state.enum = neighbor[2]
            print(child_state.enum)
            #child_state.parent = parent
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
        
        child = self.puzzle 
        i,j = self.find_zero()
        child[i][j], child[index1][index2] = child[index1][index2], child[i][j]
        return child

    def getDistance(self):
        return self.gdistance + self.hdistance

    def heuristic(self):
        distance = 0

        return distance

def BFS(puzzle):
    """
    Breadth-First Search.

    Arguments:
    - puzzle: Node object representing initial state of the puzzle

    Return:
    final_solution: An ordered list of moves representing the final solution.
    """
    final_solution = []

    puzzle=Puzzle(puzzle)

    visited = []
    queue = [puzzle]
    while queue:
        state = queue.pop(0)
        if state.puzzle == goal:
            break
        if state not in visited:
            visited.append(state)
            for child in state.child_states():
                queue.append(child)



    for node in visited:
        
        final_solution.append(node.enum)
    
    return final_solution


def DFS(puzzle):
    """
    Depth-First Search.

    Arguments:
    - puzzle: Node object representing initial state of the puzzle

    Return:
    final_solution: An ordered list of moves representing the final solution.
    """
    final_solution = []

    puzzle=Puzzle(puzzle)

    visited = set()
    stack = [puzzle]
    while stack:
        state = stack.pop()
        if state.puzzle == goal:
            #######
            break
        if state not in visited:
            visited.add(state)
            for child in state.child_states():
                stack.append(child)
        

    # TODO: WRITE CODE

    return final_solution


def A_Star_H1(puzzle):
    """
    A-Star with Heuristic 1

    Arguments:
    - puzzle: Node object representing initial state of the puzzle

    Return:
    final_solution: An ordered list of moves representing the final solution.
    """

    puzzle=Puzzle(puzzle)






    final_solution = []

    # TODO: WRITE CODE

    return final_solution


def A_Star_H2(puzzle):
    """
    A-Star with Heauristic 2

    Arguments:
    - puzzle: Node object representing initial state of the puzzle

    Return:
    final_solution: An ordered list of moves representing the final solution.
    """

    puzzle=Puzzle(puzzle)





    final_solution = []

    # TODO: WRITE CODE

    return final_solution








def read_puzzle(filename):
        """
        Helper to read a puzzle from a file.

        Arguments:
            filename: Name of file to read from.
        """
        puzzle = []
        with open(filename, "r") as f:
            for line in f.readlines():
                puzzle.append(line.split(' '))
        return puzzle

#board = read_puzzle('test_data/ex1.txt')
board=[[1, 2, 3], [0, 5, 6], [4, 7, 8]]
print(board)

final_solution = []

puzzle=Puzzle(board)

children = puzzle.child_states()

for child in children:
    print(child.puzzle)