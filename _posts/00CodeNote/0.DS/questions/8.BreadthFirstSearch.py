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
                currentVert.setDistance(currentVert.pred.getDistance + 1)
                print("currentVert.getDistance", currentVert.getDistance)
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


g = GraphyAL()
g.addEdge("1", "2", 10)
g.addEdge("1", "3", 15)
g.addEdge("2", "5", 7)
g.addEdge("2", "4", 7)
g.addEdge("3", "5", 15)
g.addEdge("4", "5", 10)
g.addEdge("4", "6", 7)
g.addEdge("5", "6", 13)

# g.addEdge('A', 'B')
# g.addEdge('A', 'C')
# g.addEdge('A', 'D')
# g.addEdge('A', 'E')
# g.addEdge('B', 'G')
# g.addEdge('F', 'G')
# g.addEdge('E', 'F')
# g.addEdge('D', 'H')
# g.addEdge('H', 'F')
g.print_list()

bfs(g.vertList["1"])

# for i in g.vertList['2'].getConnections():
#     print(i)
