from pythonds.graphs import Graph, Vertex


# The knightGraph function makes one pass over the entire board.
def knightGraph(bdSize):
    ktGraph = Graph()
    for row in range(bdSize):
        for col in range(bdSize):
            # At each square on the board the knightGraph function calls a helper, genLegalMoves,
            # to create a list of legal moves for that position on the board.
            # # All legal moves are then converted into edges in the graph.
            newPositions = genLegalMoves(row, col, bdSize)
            nodeId = posToNodeId(row, col, bdSize)
            for e in newPositions:
                nid = posToNodeId(e[0], e[1], bdSize)
                ktGraph.addEdge(nodeId, nid)
    return ktGraph


# Another helper function posToNodeId converts a location on the board in terms of a row and a column into a linear vertex number similar to the vertex numbers


def posToNodeId(row, column, board_size):
    return (row * board_size) + column


# The genLegalMoves takes the position of the knight on the board and generates each of the eight possible moves.
def genLegalMoves(x, y, bdSize):
    newMoves = []
    moveOffsets = [
        (-1, -2),
        (-1, 2),
        (-2, -1),
        (-2, 1),
        (1, -2),
        (1, 2),
        (2, -1),
        (2, 1),
    ]
    for i in moveOffsets:
        newX = x + i[0]
        newY = y + i[1]
        if legalCoord(newX, bdSize) and legalCoord(newY, bdSize):
            newMoves.append((newX, newY))
    return newMoves


# The legalCoord helper function makes sure that a particular move that is generated is still on the board.


def legalCoord(x, bdSize):
    if x >= 0 and x < bdSize:
        return True
    else:
        return False


def knightTour(n, path, u, limit):
    # n, the current depth in the search tree; the path goal
    # path, a list of vertices visited up to this point;
    # u, the vertex in the graph we wish to explore;
    # limit the number of nodes in the path.
    u.setColor("gray")
    path.append(u)

    # checks the base case condition.
    if n < limit:

        # If the path is not long enough
        # continue to explore one level deeper by choosing a new vertex to explore
        # and calling knightTour recursively for that vertex.
        nbrList = list(u.getConnections())
        i = 0
        done = False

        while i < len(nbrList) and not done:
            # DFS also uses colors to keep track of which vertices in the graph have been visited.
            # Unvisited vertices are colored white,
            # visited vertices are colored gray.
            if nbrList[i].getColor() == "white":
                done = knightTour(n + 1, path, nbrList[i], limit)
            i = i + 1
        # backtrack
        # If all neighbors of a particular vertex have been explored and we have not yet reached our goal length of 64 vertices, we have reached a dead end.
        # When we reach a dead end we must backtrack.
        # Backtracking happens when we return from knightTour with a status of False.
        if not done:
            path.pop()
            u.setColor("white")

    # If we have a path that contains 64 vertices
    # return from knightTour with a status of True,
    # indicating that we have found a successful tour.
    else:
        done = True
    return done


# In the listing below we show the code that speeds up the knightTour. This function will be used in place of the call to u.getConnections in the code previously shown above.
def orderByAvail(n):
    resList = []
    for v in n.getConnections():
        if v.getColor() == "white":
            c = 0
            for w in v.getConnections():
                if w.getColor() == "white":
                    c = c + 1
            resList.append((c, v))
    resList.sort(key=lambda x: x[0])
    return [y[1] for y in resList]


class DFSGraph(Graph):
    def __init__(self):
        super().__init__()
        self.time = 0

    def dfs(self):
        # iterates over all of the vertices in the graph
        # calling dfsvisit on the nodes that are white.
        for aVertex in self:
            aVertex.setColor("white")
            aVertex.setPred(-1)
        for aVertex in self:
            if aVertex.getColor() == "white":
                self.dfsvisit(aVertex)

    def dfsvisit(self, startVertex):
        startVertex.setColor("gray")
        self.time += 1

        startVertex.setDiscovery(self.time)

        for nextVertex in startVertex.getConnections():
            # explores all of the neighboring white vertices as deeply as possible.
            if nextVertex.getColor() == "white":
                nextVertex.setPred(startVertex)
                self.dfsvisit(nextVertex)

        startVertex.setColor("black")
        self.time += 1
        startVertex.setFinish(self.time)
