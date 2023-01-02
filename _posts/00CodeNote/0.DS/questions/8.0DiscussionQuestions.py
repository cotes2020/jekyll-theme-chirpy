#  DsLearn


# 8.25. Discussion Questions


# -------------------------------------- Excercises -------------------------------------------------
# 1. Draw the graph corresponding to the following adjacency matrix.
class Vertex:
    def __init__(self, key) -> None:
        self.id = key
        self.connectedTo = {}
        self.color = "white"
        self.distance = 0
        self.pred = None

    def __str__(self) -> str:
        return str(self.id) + " connectedTo: " + str([x.id for x in self.connectedTo])

    def addNeighbor(self, nbr, weight=0):
        self.connectedTo[nbr] = weight

    def getConnections(self):
        return self.connectedTo.keys()

    def getId(self):
        return self.id

    def getWeight(self, nbr):
        return self.connectedTo[nbr]

    def setColor(self, color):
        self.color = color
        return self.color

    def getColor(self):
        return self.color

    def setDistance(self, number):
        self.distance = number
        return self.distance

    def getDistance(self):
        return self.distance

    def setPred(self, pred):
        self.pred = pred
        return self.pred


class GraphyAM:
    def __init__(self) -> None:
        self.vertList = {}
        self.numVertices = 0
        # #adjacency matrix of size 10x10 initialize with 0
        self.am = [[0 for column in range(6)] for row in range(6)]

    def __iter__(self):
        return iter(self.vertList.values())

    def __contains__(self, n):
        return n in self.vertList

    def getIndex(self, key):
        if key not in self.vertList.keys():
            print(f"Vertex {key} not present in Graph.")
        else:
            newVertex = self.vertList[key]
        return newVertex

    # {"id":vertex}
    def addVertex(self, key):
        if key not in self.vertList.keys():
            print(
                f"Vertex {key} not present in Graph, adding it automatically."
            )
            newVertex = Vertex(key)
            self.vertList[key] = newVertex
            self.numVertices += 1
        else:
            newVertex = self.vertList[key]
        return newVertex

    def getVertex(self, n):
        if n in self.vertList:
            return self.vertList[n]
        else:
            return None

    def getVertices(self):
        # returns the names of all of the vertices in the graph
        return self.vertList.keys()

    def addEdge(self, f, t, weight=0):
        if f not in self.vertList:
            self.addVertex(f)
        if t not in self.vertList:
            self.addVertex(t)
        # action on the Vertex property
        fvert = self.getVertex(f)
        tvert = self.getVertex(t)
        self.vertList[f].addNeighbor(tvert, weight)
        # for index
        n = 0
        indexF = 0
        indexT = 0
        print(fvert.id, tvert.id)
        for key in self.vertList.keys():
            if fvert.id == key:
                indexF = n
            if tvert.id == key:
                indexT = n
            n += 1
        print("indexF", indexF, "indexT", indexT)
        self.am[indexT][indexF] = weight

    def print_graph(self):
        print("\n")
        name_str = ""
        for key in self.vertList.keys():
            name_str += key + " ,"
        name_list = name_str.replace(" ,", "")
        print(name_list)
        row = 0
        print("  ", name_str)
        for i in self.am:
            # print(row)
            if row < self.numVertices:
                print(name_list[row], i)
                row += 1
            else:
                print("0", i)


# g = GraphyAM()
# # g.addVertex('A')
# # g.addVertex('B')
# # g.addVertex('C')
# # g.addVertex('D')
# # g.addVertex('E')
# # g.addVertex('F')
# g.addEdge('A', 'B', 2)
# g.addEdge('A', 'D', 1)
# g.addEdge('A', 'E', 6)

# g.addEdge('B', 'A', 7)
# g.addEdge('B', 'C', 2)
# g.addEdge('B', 'F', 1)

# g.addEdge('C', 'A', 5)

# g.addEdge('D', 'B', 7)
# g.addEdge('D', 'E', 5)

# g.addEdge('E', 'B', 3)
# g.addEdge('E', 'D', 2)
# g.addEdge('E', 'F', 8)

# g.addEdge('F', 'A', 1)
# g.addEdge('F', 'C', 8)
# g.addEdge('F', 'D', 4)

# g.print_graph()


# -------------------------------------- Excercises -------------------------------------------------
# 2. Draw the graph corresponding to the following list of edges.


class GraphyAL:
    def __init__(self) -> None:
        self.vertList = {}
        self.numVertices = 0

    def __iter__(self):
        return iter(self.vertList.values())

    def __contains__(self, n):
        return n in self.vertList

    # {"id":vertex}
    def addVertex(self, key):
        if key not in self.vertList.keys():
            print(
                f"Vertex {key} not present in Graph, adding it automatically."
            )
            newVertex = Vertex(key)
            self.vertList[key] = newVertex
            self.numVertices += 1
        return self.vertList[key]

    def getVertex(self, n):
        if n in self.vertList.keys():
            return n
        else:
            return None

    def getVertices(self):
        # returns the names of all of the vertices in the graph
        return self.vertList.keys()

    def addEdge(self, f, t, weight=0):
        if f not in self.vertList.keys():
            fvert = self.addVertex(f)
        if t not in self.vertList.keys():
            tvert = self.addVertex(t)
        fvert = self.vertList[f]
        tvert = self.vertList[t]
        fvert.addNeighbor(tvert, weight)

    def print_list(self):
        print(self.vertList.keys())

        print("From    To      Cost")

        for id in self.vertList.keys():
            vert = self.vertList[id]
            connectList = vert.connectedTo
            for i in connectList:
                print(id, "     ", i.id, "     ", vert.getWeight(i))


# g = GraphyAL()
# g.addEdge('1', '2', 10)
# g.addEdge('1', '3', 15)
# g.addEdge('2', '3', 7)
# g.addEdge('3', '4', 7)
# g.addEdge('3', '6', 10)
# g.addEdge('4', '5', 7)
# g.addEdge('6', '4', 5)
# g.addEdge('1', '6', 5)
# g.addEdge('5', '6', 13)
# g.print_list()
# print(g.vertList)


# -------------------------------------- Excercises -------------------------------------------------
# 3. Ignoring the weights,
# perform a breadth first search on the graph from the previous question.

# from pythonds.graphs import Graph, Vertex
from pythonds.basic import Queue


def bfs(start):
    vertQueue = Queue()

    # add the first
    vertQueue.enqueue(start)
    outputstr = []

    while vertQueue.size() > 0:

        # remove it from Q
        currentVert = vertQueue.dequeue()

        # process if not seen
        if currentVert.getColor() == "white":
            currentVert.setColor("gray")
            print("=========output", currentVert.id)
            outputstr.append(currentVert.id)

            if currentVert != start:
                currentVert.setDistance(currentVert.pred.distance + 1)
                print("currentVert.distance", currentVert.distance)
                print("currentVert.pred", currentVert.pred.id)

            # add unseen child
            for nbr in currentVert.getConnections():
                if nbr.getColor() == "white":
                    print("-----nbr.id: ", nbr.id, "not seen yet")
                    nbr.setPred(currentVert)
                    vertQueue.enqueue(nbr)
                else:
                    print("have seen:", nbr.id)
            currentVert.setColor("black")
            print(currentVert.id, currentVert.getColor())
    print(outputstr)


# g = GraphyAL()
# g.addEdge('1', '2', 10)
# g.addEdge('1', '3', 15)
# g.addEdge('2', '5', 7)
# g.addEdge('2', '4', 7)
# g.addEdge('3', '5', 15)
# g.addEdge('4', '5', 10)
# g.addEdge('4', '6', 7)
# g.addEdge('5', '6', 13)

# # g.addEdge('A', 'B')
# # g.addEdge('A', 'C')
# # g.addEdge('A', 'D')
# # g.addEdge('A', 'E')
# # g.addEdge('B', 'G')
# # g.addEdge('F', 'G')
# # g.addEdge('E', 'F')
# # g.addEdge('D', 'H')
# # g.addEdge('H', 'F')
# # g.print_list()

# bfs(g.vertList['1'])

# for i in g.vertList['2'].getConnections():
#     print(i)


# -------------------------------------- Excercises -------------------------------------------------
# 4. What is the Big-O running time of the buildGraph function?
# B. O(n^2)


# -------------------------------------- Excercises -------------------------------------------------
# 5. Derive the Big-O running time for the topological sort algorithm.


# -------------------------------------- Excercises -------------------------------------------------
# 6. Derive the Big-O running time for the strongly connected components algorithm.


# -------------------------------------- Excercises -------------------------------------------------
# 7. Show each step in applying Dijkstra’s algorithm to the graph shown above.
from pythonds.graphs import Graph, PriorityQueue, Vertex


def dijkstra(aGraph, start):
    pq = PriorityQueue()
    start.setDistance(0)
    pq.buildHeap([(v.getDistance(), v) for v in aGraph])
    while not pq.isEmpty():
        currentVert = pq.delMin()
        print(currentVert.id)
        for nextVert in currentVert.getConnections():
            newDist = currentVert.getDistance() + currentVert.getWeight(nextVert)
            if newDist < nextVert.getDistance():
                nextVert.setDistance(newDist)
                nextVert.setPred(currentVert)
                pq.decreaseKey(nextVert, newDist)


g = GraphyAL()
g.addEdge("1", "2", 10)
g.addEdge("1", "3", 15)
g.addEdge("2", "5", 6)
g.addEdge("2", "4", 7)
g.addEdge("3", "5", 17)
g.addEdge("4", "5", 10)
g.addEdge("4", "6", 3)
g.addEdge("5", "6", 13)
g.print_list()

dijkstra(g, g.vertList["1"])

# for i in g:
#     print(i.id)
print([(v.getDistance(), v.id) for v in g])


import sys

# -------------------------------------- Excercises -------------------------------------------------
# 8. Using Prim’s algorithm,
# find the minimum weight spanning tree for the graph shown above.
from pythonds.graphs import Graph, PriorityQueue, Vertex


def prim(G, start):
    pq = PriorityQueue()
    for v in G:
        v.setDistance(sys.maxsize)
        v.setPred(None)
    start.setDistance(0)
    pq.buildHeap([(v.getDistance(), v) for v in G])

    while not pq.isEmpty():
        currentVert = pq.delMin()
        for nextVert in currentVert.getConnections():
            newCost = currentVert.getWeight(nextVert)
            if nextVert in pq and newCost < nextVert.getDistance():
                nextVert.setPred(currentVert)
                nextVert.setDistance(newCost)
                pq.decreaseKey(nextVert, newCost)


# -------------------------------------- Excercises -------------------------------------------------
# Draw a dependency graph illustrating the steps needed to send an email. Perform a topological sort on your graph.


# -------------------------------------- Excercises -------------------------------------------------
# Derive an expression for the base of the exponent used in expressing the running time of the knights tour.


# -------------------------------------- Excercises -------------------------------------------------
# 11. Explain why the general DFS algorithm is not suitable for solving
# the knights tour problem.

# - The knight’s tour is a special case of a depth first search where the goal is to create the deepest depth first tree, without any branches.

# The more general depth first search is actually easier.
# - Its goal is to search as deeply as possible, connecting as many nodes in the graph as possible and branching where necessary.


# 12. What is the Big-O running time for Prim’s minimum
# spanning tree algorithm?
# A. O(1)
# B. O(n^3)
# C. O(n) - The time it takes for this program to run doesn't grow linearly.
# D. O(n^2)
