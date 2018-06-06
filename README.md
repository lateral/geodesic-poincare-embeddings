# Poincaré Embeddings with "geodesic" updates

This is an adaption Nickel &amp; Kiela's [Poincaré Embeddings for Learning Hierarchical Representations](https://papers.nips.cc/paper/7213-poincare-embeddings-for-learning-hierarchical-representations) to use the exponential map (instead of the retraction) for gradient descent, thereby properly respecting the hyperbolic geometry.
For an explanation of the difference, see [Gradient descent in hyperbolic space](https://arxiv.org/abs/1805.08207).

We've previously published [an alternative implementation](https://github.com/lateral/poincare-embeddings) that offers only the retraction update.  This implementation differs in that it uses the **hyperboloid model** (instead of the Poincaré model) where the computation of the exponential map is [very easy](https://arxiv.org/abs/1805.08207).  On the other hand, here we use locking instead of HogWild.

Only the objective for embedding taxonomies is implemented.

## Requirements

For evaluation of the embeddings, you'll need the Python 3 library scikit-learn.

## Installation

```
git clone git@github.com:lateral/geodesic-poincare-embeddings.git
cd geodesic-poincare-embeddings
cd build
cmake ../
make
```

## Usage

```
$ ./poincare
    -graph                      training file path
    -output-vectors             file path for trained vectors
    -input-vectors              file path for init vectors (optional)
    -retraction-updates         use the retraction updates of Nickel & Kiela (0 or 1) [0]
    -verbose                    print progress meter (0 or 1) [0]
    -start-lr                   start learning rate [0.05]
    -end-lr                     end learning rate [0.05]
    -dimension                  manifold dimension [100]
    -max-step-size              max. hyperbolic distance of an update [2]
    -init-std-dev               stddev of the hyperbolic distance from the base point for initialization [0.1]
    -epochs                     number of epochs [5]
    -number-negatives           number of negatives sampled [5]
    -distribution-power         power used to modified distribution for negative sampling [1]
    -checkpoint-interval        save vectors every this many epochs [-1]
    -threads                    number of threads [1]
    -seed                       seed for the random number generator [1]
                                  n.b. only deterministic if single threaded!
```

### Example

```
./poincare -graph ../wordnet/mammal_closure.tsv -number-negatives 20 -epochs 50 -output-vectors vectors.csv -start-lr 0.5 -end-lr 0.5
```

### Burn-in
To achieve burn-in, just train twice, initialising the second time with the vectors trained during the first time (i.e. during burn-in).  For example:

```
./poincare -graph ../wordnet/mammal_closure.tsv -number-negatives 2 -epochs 40 -output-vectors vectors-after-burnin.csv -start-lr 0.005 -end-lr 0.005 -distribution-power 1
./poincare -graph ../wordnet/mammal_closure.tsv -number-negatives 20 -epochs 500 -input-vectors vectors-after-burnin.csv -output-vectors vectors.csv -start-lr 0.5 -end-lr 0.5 -distribution-power 0
```

### Retraction vs exponential map updates

By default, this implementation uses the exponential map to update each point along the geodesic ray defined by the (negative of the) gradient vector.  The original [implementation of Nickel and Kiela](https://github.com/facebookresearch/poincare-embeddings) used instead the retraction updates, which only approximate the exponential map.
You can train using the retraction updates by specifying the command line argument `-retraction-updates 1`.

For more information on retractions and the exponential map, see [Gradient descent in hyperbolic space](https://arxiv.org/abs/1805.08207).

## Training data

Training data is a two-column tab-separated CSV file without header.  The training files for the  WordNet hypernymy hierarchy and its mammal subtree and included in the `wordnet` folder.  These were derived as per the [implementation of the authors](https://github.com/facebookresearch/poincare-embeddings).

## Output format

Vectors are written out as a spaced-separated CSV without header, where the first column is the name of the node.

```
sen.n.01 -0.07573256650403173837 0.04804740830803629381
unit_of_measurement.n.01 -0.3194984358578525614 0.5269294142957902365
chorionic_villus_sampling.n.01 -0.1497520758532252668 0.01760349013420301248
assay.n.04 -0.3628120882612646686 0.05198792878325033239
egyptian.n.01 0.1210250472607995836 -0.01964832136051103934
...
```

Note that the vectors are points in the Poincaré ball model (even though the hyperboloid model is used during training).

## Evaluation

The script `evaluate` measures the performance of the trained embeddings:

```
$ ./evaluate --graph wordnet/noun_closure.tsv --vectors build/vectors.csv --sample-size 1000 --sample-seed 2 --include-map
Filename: build/vectors.csv
Random seed: 2
Using a sample of 1000 of the 82115 nodes.
65203 vectors are on the boundary (they will be pulled back).
mean rank:               286.23
mean precision@1:        0.3979
mean average precision:  0.5974
```

The mean average precision is not calculated by default since it is quite slow (you can turn it on with the option `--include-map`.  The precision@1 is calculated as a proxy (it's faster).

## Locking and multi-threading

HogWild allows multiple threads to simulaneously read and write common parameter vectors.  Thus "dirty reads" can occur, where the parameter vector that is read has only being partially updated by another thread.  This is often unproblematic for unconstrained optimisation, and appears to be unproblematic in practice when using the Poincaré ball model in particular.  In this implementation, however, the hyperboloid model of hyperbolic space is used (since it is easy to compute the exponential map there).  As this is constrained optimisation (points may not leave the hyperboloid), dirty reads would be catastrophic.

In order to prevent dirty reads, a locking mechanism is used.  Each parameter vector has a lock.  A thread attempts to obtain the locks of the positive sample and the negative samples for the edge it is considering.  If any of these locks can not be obtained, then this edge is skipped.  The number of edges skipped is reported for thread 0 in the console output.  Note that this means that the number of times each edge is considered during training will depend on the number of threads!

A future implementation might eliminate the need for locking by storing the parameter vectors on the Poincaré ball and performing all the intermediate hyperboloid computations using temporary variables.
