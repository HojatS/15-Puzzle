from search import BFS, DFS, A_Star_H1, A_Star_H2, UP, DOWN, LEFT, RIGHT,


def get_move_string(moves):
    """
    Helper function to print moves.

    """
    move_string = ""
    if len(moves) == 0:
        return "NONE"
    for move in moves:
        if move == UP:
            move_string = move_string + "U "
        elif move == LEFT:
            move_string = move_string + "L "
        elif move == DOWN:
            move_string = move_string + "D "
        elif move == RIGHT:
            move_string = move_string + "R "
        else:
            move_string = move_string + "INVALID "
    return move_string

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

def run_test(size, filename):
    """
    Takes a filename,
    then creates a puzzle,
    reads it in from file,
    and runs each of the search algorithms

    """
    print(f"{filename}:")
    puzzle = read_puzzle(filename)
    moves = BFS(puzzle)
    print("  BFS | " + get_move_string(moves))

    """

    puzzle = read_puzzle(filename)
    moves = DFS(puzzle)
    print("  DFS | " + get_move_string(moves))

    puzzle = read_puzzle(filename)
    moves = A_Star_H1(puzzle)
    print("  A* H1 | " + get_move_string(moves))

    puzzle = read_puzzle(filename)
    moves = A_Star_H2(puzzle)
    print("  A* H2 | " + get_move_string(moves))

    """

    print("-" * 20)


run_test(3, 'test_data/ex1.txt')
run_test(3, 'test_data/ex2.txt')
run_test(3, 'test_data/ex3.txt')
