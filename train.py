import networkx as nx
import random
import bisect
import itertools

def load_edgelist(f):
    return nx.read_edgelist(f,  delimiter='\t')

def random_walk(graph, start_node=None, size=-1, metropolized=False):
    """
    From http://www.minasgjoka.com/2.5K/sampling3.py
    random_walk(G, start_node=None, size=-1):

    Generates nodes sampled by a random walk (classic or metropolized)

    Parameters
    ----------
    graph:        - networkx.Graph
    start_node    - starting node (if None, then chosen uniformly at random)
    size          - desired sample length (int). If -1 (default), then the generator never stops
    metropolized  - False (default): classic Random Walk
                    True:  Metropolis Hastings Random Walk (with the uniform target node distribution)
    """

    if start_node==None:
        start_node = random.choice(graph.nodes())

    v = start_node
    for c in itertools.count():
        if c==size:  return
        if metropolized:   # Metropolis Hastings Random Walk (with the uniform target node distribution)
            candidate = random.choice(graph.neighbors(v))
            v = candidate if (random.random() < float(graph.degree(v))/graph.degree(candidate)) else v
        else:              # classic Random Walk
            v = random.choice(graph.neighbors(v))

        yield v

def as_spanning_trees(G):
    """
    For a given graph with multiple sub graphs, find the components
    and draw a minimum distance spanning tree.

    Returns a new Graph with cycles removed.

    Parameters
    ---------
    G:        - networkx.Graph
    """

    G2 = nx.Graph()
    # We find the connected constituents of the graph as subgraphs
    graphs = nx.connected_component_subgraphs(G, copy=False)

    # For each of these graphs we extract the minimum distance spanning tree, removing the cycles
    for g in graphs:
        T = nx.minimum_spanning_tree(g)
        G2.add_edges_from(T.edges())
        G2.add_nodes_from(T.nodes())

    return G2

def walks(G, n_walks=5, **kwargs):
    """
    A generator of random walks on a graph G.
    Takes an optional argument n_walks for traversing the same node n times.
    Traverses each node in the graph, yielding a vector of nodes of the random walk.

    """
    nodes = G.nodes()
    for n in range(n_walks):
        random.shuffle(nodes)
        for n in nodes:
            print n
            walk = random_walk(G, start_node=n, **kwargs)
            yield list(walk)
