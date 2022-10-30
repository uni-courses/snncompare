# networkx-to-lava-nc

This module compares SNN algorithms to their default/Neumann implementations.
The user can specify an SNN and "normal" algorithm which take as input a 
networkx graph, and compute some graph property as output. The output of the
SNN is then compared to the "normal" algorithm as "ground truth", in terms of:
 - Score: How many results the SNN algorithm computed correctly (from a set of input 
 graphs).
 - Runtime
In theory, the score should always be 100% for the SNN, as it should be an 
exact SNN implementation of the ground truth algorithm. This comparison is 
mainly relevant for the additions of brain adaptation and simulated radiation.

## Brain adaptation
For each SNN algorithm that the user specifies, the user can also specify a 
form of brain-inspired adaptation. This serves to increase the robustness of 
the SNN against radiation effects. The brain adaptation can be called from a
separate pip package called: `adaptivesnn`.

## Radiation
A basic form of radiation effects is modelled on the SNNs. For example, 
radiation is modelled as yielding permanent activity termination for random
neurons. 

It is noted that the accuracy of the modelling of the neuronal effects
induced by the radiation is a function of the underlying hardware platforms. 
For example, on the Intel Loihi chips, the memory/routing and computations
are somewhat intertwined from what I understood. This would suggest that
radiation effects may yield errors that prevent a computation being executed
at all, instead of a computation being corrupted, if for example a memory 
address is corrupted. (If that memory, for example, were to orchestrate some
group of neurons to do something, but instead orchestrates an inactive set of
neurons to perform some computation). In such cases, "neuronal- & synaptic"
adaptation could be the best in the world, but nothing would happen with it if
the neurons don't get the right input/send the output to the wrong place.

In hardware platforms where neurons and synapses have a more fysical 
implementation on chip, the adaptation may be more effective to increase the
radiation robustness.

## Backends
Since the effectiveness of the adaptation mechanisms, in terms of radiation
robustness, is a function of neuromorphic hardware platform, multiple backends
are supported. These backends also allow for different neuronal and synaptic
models. Currently the following backends are supported:
 - A self-made networkx SNN simulator (LIF-neurons)
 - Lava-nc simulator v0.5.0 (LIF-neurons)

## Algorithms
Different SNN implementations may use different encoding scemes, such as
sparse coding, population coding and/or rate coding. In population coding,
adaptation may be realised in the form of larger populations, whereas in rate
coding, adaptation may be realised through varying the spike-rate. This implies
that different algorithms may benefit from different types of adaptation. 
Hence, an overview is included of the implemented SNN algorithms and their 
respective compatabilities with adaptation and radiation implementations:

| Algorithm                            | Encoding | Adaptation | Radiation    |
|--------------------------------------|----------|------------|--------------|
| Minimum Dominating Set Approximation | Sparse   | Redundancy | Neuron Death |
|                                      |          |            |              |
|                                      |          |            |              |

### Minimum Dominating Set Approximation
This is an implementation of the distributed algorithm presented by Alipour et al.
- *Input*: Non-triangle, planar Networkx graph. (Non triangle means there 
should not be any 3 nodes that are all connected with eachother (forming a 
triangle)). Planar means that if you lay-out the graph on a piece of paper, no 
lines intersect (that you can roll it out on a 2D plane).
- *Output*: A set of nodes that form a dominating set in the graph.

*Description:* The algorithm basically consists of `k` rounds, where you can 
choose `k` based on how accurate you want the approximation to be, more rounds
(generally) means more accuracy. At the start each node `i` gets 1 random 
number `r_i`. This is kept constant throughout the entire algorithm. Then for
the first round:
 - Each node `i` computes how many neigbours (degree) `d_i` it has.
 - Then it adds `r_i+d_i=w_i`.
 In all consecutive rounds:
  - Each node `i` "computes" which neighbour has the heighest weight `w_j`, and
  gives that node 1 mark/point. Then each node `i` has some mark/score `m_i`.
  Next, the weight `w_i=r_i+m_i` is computed (again) and the next round starts.
This last round is repeated untill `k` rounds are completed. At the end, the 
nodes with a non-zero mark/score `m_i` are selected to form the dominating set.
  

## Experiment Stages
The experiment generates some input graphs, the SNN algorithm, a copied SNN
with some form of adaptation, and two copies with radiation (one with-/out 
adaptation). Then it simulates those SNNs for "as long as it takes" (=implicit
in the algorithm specification), and computes the results of these 4 SNNs based
on the "ground truth" Neumann/default algorithm.

This experiment is executed in 4 stages:

Input: Experiment configuration. Which consists of:
SubInput: Run configuration within an experiment.
Stage 1: Create networkx graphs that will be propagated.
Stage 2: Create propagated networkx graphs (at least one per timestep).
Stage 3: Visaualisation of the networkx graphs over time.
Stage 4: Post-processed performance data of algorithm and adaptation
mechanism.

## Running Experiment

You can run the experiment with command (to run the experiment using the
networkx backend):

```
pip install snncompare
pip install https://github.com/a-t-0/lava/archive/refs/tags/v0.5.1.tar.gz
ulimit -n 800000
python -m src
```

And run tests with:
```
python -m pytest
```

Get help with:
```
python -m src --halp
```

This generates the graphs from the default experiment configurations, and
outputs the graphs in json format to the `results/` directory, and outputs
the graph behaviour to: `latex/Images/graphs/`.


## Test Coverage
Developers can use:

```
conda env create --file environment.yml
conda activate snncompare
ulimit -n 800000
python -m pytest
```
And the respective output is listed below, with a test coverage of `65%`.:

```
============================ test session starts ==============================
platform linux -- Python 3.8.13, pytest-7.1.2, pluggy-1.0.0
rootdir: /home/name/Downloads/networkx-to-lava-nc-main, configfile:
pyproject.toml, testpaths: tests
plugins: cov-3.0.0
collected 15 items

tests/test_cyclic_graph_propagation.py .                                 [  6%]
tests/test_get_graph.py .                                                [ 13%]
tests/test_has_recurrent_edge_lava.py .....                              [ 46%]
tests/test_has_recurrent_edge_networkx.py .....                          [ 80%]
tests/test_rand_network_generation.py ..                                 [ 93%]
tests/test_rand_network_propagation.py .                                 [100%]

---------- coverage: platform linux, python 3.8.13-final-0 -----------
Name                              Stmts   Miss  Cover
-----------------------------------------------------
src.simulation.LIF_neuron.py                   113      9    92%
src/Scope_of_tests.py                39     17    56%
src/__init__.py                       0      0   100%
src/__main__.py                      16     16     0%
src/arg_parser.py                    10     10     0%
src/convert_networkx_to_lava.py     105      3    97%
src.graph_generation.get_graph.py                     83      7    92%
src.export_results.plot_graphs.py                   27      2    93%
src/run_on_lava.py                   23      2    91%
src/run_on_networkx.py               39      0   100%
src/verify_graph_is_snn.py           36      7    81%
-----------------------------------------------------
TOTAL                               491     73    85%
Coverage XML written to file cov.xml


====================== 15 passed in 147.14s (0:02:27) =========================

```
