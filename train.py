import networkx as nx
from multiprocessing import Pool
import random
import bisect
import itertools
import sys
import logging


from argparse import ArgumentParser, FileType, ArgumentDefaultsHelpFormatter

LOG_LEVEL = logging.INFO
logging.basicConfig(level=LOG_LEVEL, format='[%(levelname)s] %(name)s %(asctime)s: %(message)s')
log = logging.getLogger(__name__)


def load_edgelist(f, delimiter=None):
    return nx.read_edgelist(f, delimiter=delimiter)

def load_adjlist(f, delimiter=None):
    return nx.read_adjlist(f, delimiter=delimiter)

def random_walk(graph, size=-1, metropolized=False, start_node=None):
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

def as_spanning_forest(G):
    """
    Return a minimum spanning tree or forest of an undirected weighted
    graph.

    A minimum spanning tree is a subgraph of the graph (a tree) with the
    minimum sum of edge weights.

    If the graph is not connected a spanning forest is constructed. A
    spanning forest is a union of the spanning trees for each connected
    component of the graph.
    """
    return nx.minimum_spanning_tree(G)

def chunks(l, n):
    """Yield successive n-sized chunks from l."""
    for i in range(0, len(l), n):
        yield l[i:i+n]

def walk(G, size, n_walks, metropolized, nodes):
    curr = []
    for i in range(n_walks):
        random.shuffle(nodes)
        for n in nodes:
            walk = random_walk(G, size=size, metropolized=metropolized, start_node=n)
            curr += [list(walk)]

    return curr

def _walk(G_size_walks_metropolized_nodes):
    log.debug("Removing cycles by transforming to spanning forest")
    return walk(*G_size_walks_metropolized_nodes)

def walks(G, workers, n_walks, size, metropolized):
    p = Pool(processes=workers)
    node_divisor = len(p._pool)*4
    node_chunks = list(chunks(G.nodes(), int(G.order()/node_divisor)))
    num_chunks = len(node_chunks)
    log.info("Generating walks with %s chunks on %s processes" % (num_chunks, workers))
    walk_sc = p.map(_walk,
                    zip([G]*num_chunks,
                        [size]*num_chunks,
                        [n_walks]*num_chunks,
                        [metropolized]*num_chunks,
                        node_chunks))

    log.info("Concatenating walks")
    result = list(itertools.chain(*walk_sc))

    # Close the pool, releasing the resources
    p.close()
    p.terminate()
    p.join()
    return result

def train(args):
    if args.use_keras:
        from word2veckeras import Word2VecKeras
        log.info("Using Keras back-end")
    else:
        from gensim.models import Word2Vec
        log.info("Using vanilla GenSim back-end")

    log.info("Reading input file")
    if args.format == "edgelist":
        G = as_spanning_forest(load_edgelist(args.input, delimiter=args.delimiter))
    elif args.format == "adjlist":
        G = as_spanning_forest(load_adjlist(args.input, delimiter=args.delimiter))
    else:
        log.error("Format must be one of edgelist or adjlist")

    log.info("Generating random walks")

    # TODO make this use some file for larger than memory data sets
    all_walks = walks(G,
                      workers=args.workers,
                      n_walks=args.walks,
                      size=args.length,
                      metropolized=args.metropolized)

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
    model.save_word2vec_format(args.output, binary=args.binary)


def main():
    p = ArgumentParser("DeepOntology",
                       formatter_class=ArgumentDefaultsHelpFormatter,
                       conflict_handler='resolve')

    p.add_argument('--input', required=True, help='Input graph file')
    p.add_argument('--output', required=True, help='Output representation file')
    p.add_argument('--delimiter', help="Delimiter used in the input file, e.g. ',' for CSV", default=None)
    p.add_argument('--format', help='Format of the input file, either edgelist or adjlist', default="edgelist")
    p.add_argument('--representation-size', default=64, type=int, help='Number of latent dimensions to learn for each node.')
    p.add_argument('--walks', help="Number of walks for each node", default=5, type=int)
    p.add_argument('--length', help="Length of the random walk on the graph", default=40, type=int)
    p.add_argument('--iter', help="Number of iteration epocs", default=5, type=int)
    p.add_argument('--window-size', help="Window size of the skipgram model", type=int, default=5)
    p.add_argument('--workers', help="Number of parallel processes", type=int, default=1)
    p.add_argument('--metropolized', help="Use Metropolis Hastings for random walk", type=bool, default=False)
    p.add_argument('--binary', help="Use the binary output format for Word2Vec", type=bool, default=True)
    p.add_argument('--seed', default=1, type=int, help='Seed for random walk generator.')

    args = p.parse_args()

    random.seed(args.seed)
    train(args)

if __name__ == "__main__":
  sys.exit(main())
