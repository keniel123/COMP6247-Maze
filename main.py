import read_maze


def main():
    read_maze.load_maze()
    print(read_maze.get_local_maze_information(6,2))

if __name__ == "__main__":
    main()