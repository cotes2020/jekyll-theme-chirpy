```python
import numpy as np
import networkx as nx
from pprint import pprint
```

### Implementation


```python
class Node:
    def __init__(self):
        self.data = np.random.randn(20)
        self.wkey = np.random.randn(20, 20)
        self.wquery = np.random.randn(20, 20)
        self.wvalue = np.random.randn(20, 20)

    def key(self):
        # what do I have
        return self.wkey @ self.data

    def query(self):
        # what am I looking for?
        return self.wquery @ self.data

    def value(self):
        # what do I publicly reveal/broadcast to others?
        return self.wvalue @ self.data
```


```python
class Graph:
    def __init__(self):
        self.nodes = [Node() for i in range(10)]
        randi = lambda: np.random.randint(len(self.nodes))
        self.edges = [[randi(), randi()] for _ in range(40)]
        self.edges = set([tuple(i) for i in self.edges])
        self.edges = [list(i) for i in self.edges]

    def run_node(self, i: int, n: Node) -> np.ndarray:
        print(f"++++ ito: {i}")
        # each node has one question
        q = n.query()
        ifroms = [ifrom for (ifrom, ito) in self.edges if ito == i]
        inputs = [self.nodes[ifrom] for ifrom in ifroms]
        print(f"\tifroms: {ifroms}")

        # each key from all connected nodes
        keys = [m.key() for m in inputs]
        scores = [k.dot(q) for k in keys]
        print(f"\tscores: {scores}")
        scores = np.exp(scores)
        scores = scores / np.sum(scores)
        print(f"\tAttention weights: {scores}")
        print(f"\tsum_i scores_i = {sum(scores)}")

        # each value from all connected nodes
        values = [m.value() for m in inputs]
        # each vector is (20,)
        updates = [s * v for s, v in zip(scores, values, strict=True)]
        # resulting update is (20,)
        update = np.array(updates).sum(axis=0)
        print()
        return update

    def run(self):
        updates = []
        for i, n in enumerate(self.nodes):
            update = self.run_node(i=i, n=n)
            updates.append(update)

        # add all
        for n, u in zip(self.nodes, updates, strict=True):
            n.data = n.data + u
```

### Results


```python
graph = Graph()
G = nx.from_edgelist(graph.edges)
print(G)
print()
print("Nodes")
print(G.nodes)
print()
print("Edge List")
pprint({k: list(dict(v).keys()) for k, v in G.adj.items()})
```

    Graph with 10 nodes and 29 edges
    
    Nodes
    [4, 0, 3, 7, 5, 9, 2, 8, 1, 6]
    
    Edge List
    {0: [4, 8, 9, 6],
     1: [6, 3, 9],
     2: [9, 2, 8, 4, 5, 3, 7],
     3: [4, 7, 8, 1, 3, 5, 9, 2],
     4: [0, 3, 5, 8, 9, 2],
     5: [4, 9, 3, 2, 8],
     6: [1, 9, 0],
     7: [3, 9, 2, 8],
     8: [3, 0, 2, 4, 8, 5, 7],
     9: [2, 5, 1, 4, 7, 3, 6, 0]}



```python
graph.run()
```

    ++++ ito: 0
    	ifroms: [4, 6]
    	scores: [-82.77388839642937, 41.91637114600668]
    	Attention weights: [7.04219965e-55 1.00000000e+00]
    	sum_i scores_i = 1.0
    
    ++++ ito: 1
    	ifroms: [9]
    	scores: [36.614881545653645]
    	Attention weights: [1.]
    	sum_i scores_i = 1.0
    
    ++++ ito: 2
    	ifroms: [9, 2, 8, 5, 7]
    	scores: [82.23362072690487, -14.201833043418045, -259.7099400673758, -87.70061982216168, -44.039480507623985]
    	Attention weights: [1.00000000e+000 1.31405810e-042 3.13183177e-149 1.57941785e-074
     1.44640169e-055]
    	sum_i scores_i = 1.0
    
    ++++ ito: 3
    	ifroms: [8, 1, 3, 9, 2]
    	scores: [-91.95561647397787, -32.29587396264722, -112.44536732987606, -21.12666640859117, 66.7936485052498]
    	Attention weights: [1.13781122e-69 9.24628292e-44 1.43709170e-78 6.55680678e-39
     1.00000000e+00]
    	sum_i scores_i = 1.0
    
    ++++ ito: 4
    	ifroms: [3, 5, 9, 2]
    	scores: [193.26582794680346, 62.00987191513143, 47.6742960396918, -99.48013171067741]
    	Attention weights: [1.00000000e+000 9.91431192e-058 5.89387717e-064 7.27855422e-128]
    	sum_i scores_i = 1.0
    
    ++++ ito: 5
    	ifroms: [9, 4, 3]
    	scores: [-80.36923024933061, -125.15411273964283, -163.97414270916659]
    	Attention weights: [1.00000000e+00 3.54954480e-20 4.90735932e-37]
    	sum_i scores_i = 1.0
    
    ++++ ito: 6
    	ifroms: [1, 9]
    	scores: [-153.75267615934996, 2.661427187515301]
    	Attention weights: [1.17548753e-68 1.00000000e+00]
    	sum_i scores_i = 1.0
    
    ++++ ito: 7
    	ifroms: [3]
    	scores: [81.34528630015134]
    	Attention weights: [1.]
    	sum_i scores_i = 1.0
    
    ++++ ito: 8
    	ifroms: [0, 2, 4, 8, 3, 5, 7]
    	scores: [-67.7028096443193, 136.6460724643095, 38.112608393001025, 209.32068976622077, -285.06617766718176, 38.686187965411314, -84.24955877048632]
    	Attention weights: [4.90030203e-121 2.74040487e-032 4.41850146e-075 1.00000000e+000
     1.95214264e-215 7.84110818e-075 3.19198623e-128]
    	sum_i scores_i = 1.0
    
    ++++ ito: 9
    	ifroms: [1, 7, 0, 2]
    	scores: [54.550609596374215, -37.4602959389309, -63.439279766739226, 54.01263056898778]
    	Attention weights: [6.31342163e-01 6.92526258e-41 3.61332414e-52 3.68657837e-01]
    	sum_i scores_i = 1.0
    

