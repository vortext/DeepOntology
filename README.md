# DeepOntology

DeepOntology is a reimplementation of [DeepWalk](https://github.com/phanein/deepwalk), specifically tuned for ontologies.
Under the hood it uses NetworkX and Gensim to construct a SkipGram model on (Metropolized) Random Walks from spanning forests.
By walking `parent-of` relations of an ontology a lower dimensional continuous embedding is made that encodes for those relations.
These embeddings can be used for clustering and classification tasks, without resorting to sparse models of ancestor relations.

## Caveat
The random walks are constructed using multi-process parallelization without shared memory, which means the whole graph has to fit it memory `n` times (n=number of processes). Out of core or shared memory realization is work in progress.

## Usage
```
python train.py --help
usage: DeepOntology [-h] --input INPUT --output OUTPUT [--delimiter DELIMITER]
                    [--format FORMAT]
                    [--representation-size REPRESENTATION_SIZE]
                    [--walks WALKS] [--length LENGTH] [--iter ITER]
                    [--window-size WINDOW_SIZE] [--workers WORKERS]
                    [--metropolized METROPOLIZED] [--binary BINARY]
                    [--seed SEED]

optional arguments:
  -h, --help            show this help message and exit
  --input INPUT         Input graph file (default: None)
  --output OUTPUT       Output representation file (default: None)
  --delimiter DELIMITER
                        Delimiter used in the input file, e.g. ',' for CSV
                        (default: None)
  --format FORMAT       Format of the input file, either edgelist or adjlist
                        (default: edgelist)
  --representation-size REPRESENTATION_SIZE
                        Number of latent dimensions to learn for each node.
                        (default: 64)
  --walks WALKS         Number of walks for each node (default: 5)
  --length LENGTH       Length of the random walk on the graph (default: 40)
  --iter ITER           Number of iteration epocs (default: 5)
  --window-size WINDOW_SIZE
                        Window size of the skipgram model (default: 5)
  --workers WORKERS     Number of parallel processes (default: 1)
  --metropolized METROPOLIZED
                        Use Metropolis Hastings for random walk (default:
                        False)
  --binary BINARY       Use the binary output format for Word2Vec (default:
                        True)
  --seed SEED           Seed for random walk generator. (default: 1)
```

## Citing
```
@inproceedings{Perozzi:2014:DOL:2623330.2623732,
 author = {Perozzi, Bryan and Al-Rfou, Rami and Skiena, Steven},
 title = {DeepWalk: Online Learning of Social Representations},
 booktitle = {Proceedings of the 20th ACM SIGKDD International Conference on Knowledge Discovery and Data Mining},
 series = {KDD '14},
 year = {2014},
 isbn = {978-1-4503-2956-9},
 location = {New York, New York, USA},
 pages = {701--710},
 numpages = {10},
 url = {http://doi.acm.org/10.1145/2623330.2623732},
 doi = {10.1145/2623330.2623732},
 acmid = {2623732},
 publisher = {ACM},
 address = {New York, NY, USA},
 keywords = {deep learning, latent representations, learning with partial labels, network classification, online learning, social networks},
}
```
