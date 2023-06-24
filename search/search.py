# search.py
# ---------------
# Licensing Information:  You are free to use or extend this projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to the University of Illinois at Urbana-Champaign
#
# Created by Kelvin Ma (kelvinm2@illinois.edu) on 01/24/2021

"""
This is the main entry point for MP1. You should only modify code
within this file -- the unrevised staff files will be used for all other
files and classes when code is run, so be careful to not modify anything else.
"""
import heapq
# Search should return the path.
# The path should be a list of tuples in the form (row, col) that correspond
# to the positions of the path taken by your search algorithm.
# maze is a Maze object based on the maze from the file specified by input filename
# searchMethod is the search method specified by --method flag (bfs,dfs,astar,astar_multi,fast)


# Feel free to use the code below as you wish
# Initialize it with a list/tuple of objectives
# Call compute_mst_weight to get the weight of the MST with those objectives
# TODO: hint, you probably want to cache the MST value for sets of objectives you've already computed...
class MST:
    def __init__(self, objectives):
        self.elements = {key: None for key in objectives}

        # TODO: implement some distance between two objectives 
        # ... either compute the shortest path between them, or just use the manhattan distance between the objectives
        self.distances   = {
                (i, j): manhattan(i, j)
                for i, j in self.cross(objectives)
            }
        
    # Prim's algorithm adds edges to the MST in sorted order as long as they don't create a cycle
    def compute_mst_weight(self):
        weight      = 0
        for distance, i, j in sorted((self.distances[(i, j)], i, j) for (i, j) in self.distances):
            if self.unify(i, j):
                weight += distance
        return weight

    # helper checks the root of a node, in the process flatten the path to the root
    def resolve(self, key):
        path = []
        root = key 
        while self.elements[root] is not None:
            path.append(root)
            root = self.elements[root]
        for key in path:
            self.elements[key] = root
        return root
    
    # helper checks if the two elements have the same root they are part of the same tree
    # otherwise set the root of one to the other, connecting the trees
    def unify(self, a, b):
        ra = self.resolve(a) 
        rb = self.resolve(b)
        if ra == rb:
            return False 
        else:
            self.elements[rb] = ra
            return True

    # helper that gets all pairs i,j for a list of keys
    def cross(self, keys):
        return (x for y in (((i, j) for j in keys if i < j) for i in keys) for x in y)

def bfs(maze):
    """
    Runs BFS for part 1 of the assignment.

    @param maze: The maze to execute the search on.

    @return path: a list of tuples containing the coordinates of each state in the computed path
    """
    queue = []
    visited = []

    queue.append([maze.start])
   
    while queue:
        f = queue.pop(0)        
        i,j = f[-1]

        cell = maze[i,j]

        if (i,j) in visited:
            continue
        visited.append((i,j))
 
        if cell == maze.legend.waypoint: 
            return f
        nb = maze.neighbors(i,j)
        for k in nb:
            if k not in visited:
                queue.append(f + [k]) 
    return []

def astar_single(maze):
    """
    Runs A star for part 2 of the assignment.

    @param maze: The maze to execute the search on.

    @return path: a list of tuples containing the coordinates of each state in the computed path
    """
    openl = []
    
    closel = {}
    
    heapq.heappush(openl,(manhattan(maze.start,maze.waypoints[0]),[maze.start]))
    
    
    while openl:
        current = heapq.heappop(openl)[1]
        i,j = current[-1]

        if (i,j) in closel:
            continue
        closel[(i,j)] = len(current) + manhattan((i,j),maze.waypoints[0])
        
        if maze[i,j] == maze.legend.waypoint:
            return current

        nb = maze.neighbors(i,j)
        for k in nb:
            cost = len(current) + manhattan(k,maze.waypoints[0])
            if k not in closel:
                heapq.heappush(openl,(cost,current+[k]))
            else:
                if closel[k] > cost:
                    closel[k] = cost
                    heapq.heappush(openl,(cost,current+[k]))
        
        
        
    return []

def manhattan(a, b):
    
    return abs(a[0] - b[0]) + abs(a[1] - b[1])

def astar_multiple(maze):
    """
    Runs A star for part 3 of the assignment in the case where there are
    multiple objectives.

    @param maze: The maze to execute the search on.

    @return path: a list of tuples containing the coordinates of each state in the computed path
    """
    openl = []
    closel = {}
    dic = {}
    start = maze.start      #getting start
    
    graph = MST(list(maze.waypoints))    #constructing mst tree
    weight = graph.compute_mst_weight() # getting initial weight of mst with all wp
    

    for x in list(maze.waypoints):       # constructing the dictionary with the key as the waypoint coordinates and value as the distance
        dic[x] = manhattan(start,x)
    wp = (maze.waypoints)
    t_cost = weight + dic[min(dic, key=dic.get)] 
    
    heapq.heappush(openl,(t_cost,[maze.start]))  #pushing the start onto the priority queue
    
    while openl:
        popped = heapq.heappop(openl)       # popping the current
        current = popped[1]
        i,j = current[-1]                       # the last cell on current list is the currently visiting cell
        
        for y in maze.waypoints:
            dic[y] = manhattan(y,(i,j))
        wp = (maze.waypoints) 

        for x in maze.waypoints:
            if x in current:
                del dic[x]            # deleting the reached waypoint
                wp = list(wp)
                wp.remove(x)
                wp = tuple(wp)
        
        if ((i,j),wp) in closel: 
            continue
        if len(dic): 
            closel[((i,j),wp)] = len(current) + weight + dic[min(dic, key=dic.get)]
        else:
            closel[((i,j),wp)] = len(current) + weight
            
        if len(wp) == 0:                
            return current
        
        
        graph = MST(dic)
        weight = graph.compute_mst_weight()
        nb = maze.neighbors(i,j)
        for k in nb:
            for x in dic:       # constructing the dictionary with the key as the waypoint coordinates and value as the distance
                dic[x] = manhattan(x,k) # finding the distance from the neighbor to each of remaining waypoints
                
            new_cost = len(current)+ weight + dic[min(dic, key=dic.get)]
            if (k,wp) not in closel:   
                heapq.heappush(openl,(new_cost,current+[k]))
            else:
                if closel[(k,wp)] > new_cost:
                    closel[(k,wp)] = new_cost
                    heapq.heappush(openl,(new_cost,current+[k]))
               
    return []
        
        
def fast(maze):
    """
    Runs suboptimal search algorithm for part 4.

    @param maze: The maze to execute the search on.

    @return path: a list of tuples containing the coordinates of each state in the computed path
    """
    openl = []
    closel = {}
    dic = {}
    ret_path = []
    start = maze.start      #getting start
    
    graph = MST(list(maze.waypoints))    #constructing mst tree
    weight = graph.compute_mst_weight() # getting initial weight of mst with all wp
    
    for x in list(maze.waypoints):       #dictionary with the key as the waypoint coordinates and value as the distance to find distance to nearest goal
        dic[x] = manhattan(start,x)
    wp = (maze.waypoints)               # initially remaining waypoints are the same as all maze waypoints
    t_cost = 2.5 * (weight + dic[min(dic, key=dic.get)]) # f = h, g = 0

    start_s = state(start[0],start[1],wp,0,None) # state consists of i,j, remaining waypoints, g, previous cell
    heapq.heappush(openl,(t_cost,start_s))  #pushing the start onto the priority queue
    
    while openl:
        popped = heapq.heappop(openl)       # popping the current
        current = popped[1]
        i = current.i                       
        j = current.j
        wp = current.wp
        g = current.g

        
        if ((i,j),wp) in closel: 
            continue
        closel[((i,j),wp)] = g
        
        if (current.i, current.j) in maze.waypoints and (current.i, current.j) in wp:
            wp = list(wp)
            wp.remove((current.i, current.j))
            wp = tuple(wp)

        if len(wp) == 0:
            ret_path.append((i,j))
            x = current.prev
            while x != None:
                ret_path.append((x.i,x.j))
                x = x.prev
            return list(reversed(ret_path))
        
        
        graph = MST(wp)
        weight = graph.compute_mst_weight()
        nb = maze.neighbors(i,j)
        for k in nb:
            minlist = []
            for x in wp:       # constructing the dictionary with the key as the waypoint coordinates and value as the distance
                minlist.append(manhattan(x,k)) # finding the distance from the neighbor to each of remaining waypoints
                
            new_cost = current.g + 2.5*(weight + min(minlist)) + 1
            if (k,wp) not in closel:
                nxt = state(k[0],k[1],wp,current.g+1,current)
                heapq.heappush(openl,(new_cost,nxt))
            else:
                if closel[(k,wp)] > current.g + 1:
                    closel[(k,wp)] = current.g + 1
                    nxt = state(k[0],k[1],wp,current.g+1,current)
                    heapq.heappush(openl,(new_cost,nxt))
               
    return []

class state:
    def __init__(self, i, j, wp, g, prev):
        self.i = i
        self.j = j
        self.wp = wp
        self.g = g
        self.prev = prev


    def __lt__(self,other):
        return self.g < other.g
            
            
