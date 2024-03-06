#  DsLearn


# 8.26. Programming Exercises
# https://runestone.academy/runestone/books/published/pythonds/Graphs/Exercises.html


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
            print(f"Vertex {key} not present in Graph, adding it automatically.")
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
            print(f"Vertex {key} not present in Graph, adding it automatically.")
            newVertex = Vertex(key)
            self.vertList[key] = newVertex
            self.numVertices += 1
        return self.vertList[key]

    def getVertex(self, n):
        if n in self.vertList.keys():
            return n
        else:
            return None

    def getVertexitem(self, n):
        if n in self.vertList.keys():
            return self.vertList[n]
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


# -------------------------------------- Exercises -------------------------------------------------
# Modify the depth first search function to produce a topological sort.
from pythonds.basic import Stack


def dfs_topo(g, vertextargetid):
    vertex_list = g.vertList
    vertextarget = vertex_list[vertextargetid]

    stack_list = Stack()
    stack_list.push(vertextarget)
    output = Stack()
    while not stack_list.isEmpty():
        currentVert = stack_list.pop()
        if currentVert.getColor() == "white":
            currentVert.setColor("grey")
            childVert = currentVert.getConnections()
            for vertexs in childVert:
                if vertexs.getColor() == "white":
                    stack_list.push(vertexs)
        currentVert.setColor("black")
        output.push(currentVert)
    for i in range(output.size()):
        print(output.pop())


# g = GraphyAL()
# g.addEdge('1', '2', 10)
# g.addEdge('2', '3', 7)
# g.addEdge('3', '4', 7)
# g.addEdge('4', '5', 7)
# g.addEdge('5', '6', 13)
# g.print_list()
# print(g.vertList)

# dfs_topo(g, '1')


# -------------------------------------- Exercises -------------------------------------------------
# Modify the depth first search to produce strongly connected components.
from pythonds.basic import Stack


def dfs_scc(g, vertextargetid):
    vertex_list = g.vertList
    vertextarget = vertex_list[vertextargetid]
    stack_list = Stack()
    stack_list.push(vertextarget)
    output = Stack()
    while not stack_list.isEmpty():
        currentVert = stack_list.pop()
        if currentVert.getColor() == "white":
            currentVert.setColor("grey")
            childVert = currentVert.getConnections()
            for vertexs in childVert:
                if vertexs.getColor() == "white":
                    stack_list.push(vertexs)
        currentVert.setColor("black")
        output.push(currentVert)
    for i in range(output.size()):
        print(output.pop())


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
# dfs_topo(g, '1')


# -------------------------------------- Exercises -------------------------------------------------
# Python program to find strongly connected components in a given
# directed graph using Tarjan's algorithm (single DFS)
# Complexity : O(V+E)
from pythonds.basic import Stack


def run(g):
    articulationPoiny_list = [False] * g.numVertices
    visitTime = [-1] * g.numVertices
    lowTime = [-1] * g.numVertices
    visited_list = []
    time = 0
    for i in range(g.numVertices):
        if visitTime[i] == -1:
            dfs_scc(
                g, i, time, visitTime, lowTime, articulationPoiny_list, visited_list
            )


def dfs_scc(g, i, time, visitTime, lowTime, articulationPoiny_list, visited_list):
    print(i)
    vertex_ids = list(g.getVertices())
    vertex_id = vertex_ids[i]
    vertex_s = g.getVertexitem(vertex_id)

    visited_list.append(vertex_s)
    visitTime[i] = time
    lowTime[i] = time
    time += 1
    v_child = vertex_s.connectedTo

    for v in v_child:
        # child is where it come from, seen before
        if v not in visited_list:
            i += 1
            dfs_scc(
                g, i, time, visitTime, lowTime, articulationPoiny_list, visited_list
            )
            v.scc_parent = vertex_s
            # elif:
            #     # meet back edge
            #             if v.parent != vertex_s:
            #                 if (vertex_s.parent == None and vertex_s.child == 2) or vertex_s.visitTime <= v.parent.lowTime:
            #                     articulationPoiny_list.append[v]
            #                 else:
            #                     vertex_s.lowTime = min(vertex_s.lowTime, v.lowTime)
            #             vertex_s.lowTime = min(vertex_s.lowTime, v.lowTime)
            #         if (vertex_s.parent == None and vertex_s.child == 2) or vertex_s.visitTime <= v.parent.lowTime:
            #                     articulationPoiny_list.append[v]


g = GraphyAL()
g.addEdge("1", "2", 10)
g.addEdge("1", "3", 15)
g.addEdge("2", "3", 7)
g.addEdge("3", "4", 7)
g.addEdge("3", "6", 10)
g.addEdge("4", "5", 7)
g.addEdge("6", "4", 5)
g.addEdge("1", "6", 5)
g.addEdge("5", "6", 13)
# print(g.numVertices)
run(g)


# solution 2
# Python program to find strongly connected components in a given
# directed graph using Tarjan's algorithm (single DFS)
# Complexity : O(V+E)

from collections import defaultdict


# This class represents an directed graph
# using adjacency list representation
class Graph:
    def __init__(self, vertices):
        # No. of vertices
        self.V = vertices

        # default dictionary to store graph
        self.graph = defaultdict(list)

        self.Time = 0

    # function to add an edge to graph
    def addEdge(self, u, v):
        self.graph[u].append(v)

    """A recursive function that find finds and prints strongly connected
    components using DFS traversal
    u --> The vertex to be visited next
    disc[] --> Stores discovery times of visited vertices
    low[] -- >> earliest visited vertex (the vertex with minimum
                discovery time) that can be reached from subtree
                rooted with current vertex
    st -- >> To store all the connected ancestors (could be part
           of SCC)
    stackMember[] --> bit/index array for faster check whether
                  a node is in stack
    """

    def SCCUtil(self, u, low, disc, stackMember, st):
        print("-----------------u:", u)
        # Initialize discovery time and low value
        disc[u] = self.Time
        low[u] = self.Time
        self.Time += 1
        stackMember[u] = True
        st.append(u)

        # Go through all vertices adjacent to this
        for v in self.graph[u]:
            print("---------v:", v)

            # If v is not visited yet, then recur for it
            if disc[v] == -1:

                self.SCCUtil(v, low, disc, stackMember, st)

                # Check if the subtree rooted with v has a connection to
                # one of the ancestors of u
                # Case 1 (per above discussion on Disc and Low value)
                low[u] = min(low[u], low[v])

            # elif stackMember[v] == True:
            #     '''Update low value of 'u' only if 'v' is still in stack
            #     (i.e. it's a back edge, not cross edge).
            #     Case 2 (per above discussion on Disc and Low value) '''
            #     low[u] = min(low[u], disc[v])
            low[u] = min(low[u], disc[v])

        print("last check")
        # head node found, pop the stack and print an SCC
        w = -1  # To store stack extracted vertices
        print("---------u:", u, "low[u]:", low[u], "disc[u]:", disc[u])
        if low[u] == disc[u]:
            while w != u:
                w = st.pop()
                print(w),
                stackMember[w] = False

            print("")

    # The function to do DFS traversal.
    # It uses recursive SCCUtil()
    def SCC(self):

        # Mark all the vertices as not visited
        # and Initialize parent and visited,
        # and ap(articulation point) arrays
        disc = [-1] * (self.V)
        low = [-1] * (self.V)
        stackMember = [False] * (self.V)
        st = []

        # Call the recursive helper function
        # to find articulation points
        # in DFS tree rooted with vertex 'i'
        for i in range(self.V):
            if disc[i] == -1:
                self.SCCUtil(i, low, disc, stackMember, st)


# Create a graph given in the above diagram
g1 = Graph(5)
g1.addEdge(1, 0)
g1.addEdge(0, 2)
g1.addEdge(2, 1)
g1.addEdge(0, 3)
g1.addEdge(3, 4)
print("SSC in first graph ")
g1.SCC()


# -------------------------------------- Exercises -------------------------------------------------


# -------------------------------------- Exercises -------------------------------------------------


# -------------------------------------- Exercises -------------------------------------------------


# -------------------------------------- Exercises -------------------------------------------------
