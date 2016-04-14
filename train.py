import networkx as nx
import random
import bisect
import itertools
import sys
import logging

from gensim.models import Word2Vec

from argparse import ArgumentParser, FileType, ArgumentDefaultsHelpFormatter

LOG_LEVEL = logging.INFO
logging.basicConfig(level=LOG_LEVEL, format='[%(levelname)s] %(name)s %(asctime)s: %(message)s')
log = logging.getLogger(__name__)


def load_edgelist(f, delimiter=None):
    return nx.read_edgelist(f,  delimiter=delimiter)

def load_adjlist(f, delimiter=None):
    return nx.read_adjlist(f,  delimiter=delimiter)

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

    log.info("Removing cycles")
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
            walk = random_walk(G, start_node=n, **kwargs)
            yield list(walk)


def train(args):
    random.seed(args.seed)
    log.info("Reading input file")
    if args.format == "edgelist":
        G = as_spanning_trees(load_edgelist(args.input, delimiter=args.delimiter))
    elif args.format == "adjlist":
        G = as_spanning_trees(load_adjlist(args.input, delimiter=args.delimiter))
    else:
        print("Format must be one of edgelist or adjlist")

    log.info("Generating random walks")
    # TODO make this use some file for larger than memory data sets
    all_walks = list(walks(G, n_walks=args.walks, size=args.length, metropolized=args.metropolized))

    log.info("Training")
    model = Word2Vec(all_walks,
                     size=args.representation_size,
                     window=args.window_size,
                     min_count=0,
                     workers=args.workers,
                     iter=args.iter,
                     sg=1,
                     trim_rule=None)

    log.info("Saving model")
    model.save_word2vec_format(args.output)


def main():
    p = ArgumentParser("DeepOntology",
                       formatter_class=ArgumentDefaultsHelpFormatter,
                       conflict_handler='resolve')

    p.add_argument('--input', required=True, help='Input graph file')
    p.add_argument('--output', required=True, help='Output representation file')
    p.add_argument('--delimiter', help="Delimiter used in the input file, e.g. ',' for CSV", default=None)
    p.add_argument('--format', help='Format of the input file, either edgelist or adjlist', default="edgelist")
    p.add_argument('--representation-size', default=64, type=int, help='Number of latent dimensions to learn for each node.')
    p.add_argument('--walks', help="Number of walks for each node", default=10, type=int)
    p.add_argument('--length', help="Length of the random walk on the graph", default=40, type=int)
    p.add_argument('--iter', help="Number of iteration epocs", default=5, type=int)
    p.add_argument('--window-size', help="Window size of the skipgram model", type=int, default=5)
    p.add_argument('--workers', help="Number of parallel processes", type=int, default=1)
    p.add_argument('--metropolized', help="Use Metropolize Hastings for random walk", type=bool, default=False)
    p.add_argument('--seed', default=1, type=int, help='Seed for random walk generator.')



    args = p.parse_args()

    train(args)

if __name__ == "__main__":
  sys.exit(main())
