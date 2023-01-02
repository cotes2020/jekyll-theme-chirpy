class Vertex:
    def __init__(self, key) -> None:
        self.id = key
        self.connectedTo = {}
        self.color = "white"
        self.getDistance = 0
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
        self.getDistance = number
        return self.getDistance

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
