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
