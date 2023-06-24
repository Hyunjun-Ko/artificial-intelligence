# artificial-intelligence
The projects are written in python3 and while the core functionalities have been written by myself the templates and the environment have been configured by and originated from University of Illinois at Urbana-Champaign. 

## Search

In .search.py,
def bfs(maze):
def astar_single(maze):
def astar_multiple(maze):
def fast(maze):


This project aims to solve mazes by using different search algorithms
1. Breadth-first search, with one waypoint.
2. A* search, with one waypoint.
3. A* search, with many waypoints.
4. Faster A* search, with many waypoints

The _data_/ directory contains the maze files. Each maze is a simple plaintext file. 

The main.py file is the primary entry point and with 

**python3 main.py --human data/part-1/small** 

it will open a pygame-based interactive visualization of the _data/part-1/small_ maze. 

The blue dot represents the agent. agent can be moved using the arrow keys. The black dots represent the maze waypoints. The agent has to go through a path that reaches all of the waypoints.