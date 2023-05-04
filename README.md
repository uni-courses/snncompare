# Spiking Neural Network Performance Tool

[![Python 3.10](https://img.shields.io/badge/python-3.10-blue.svg)](https://www.python.org/downloads/release/python-3106/)
[![License: AGPL v3](https://img.shields.io/badge/License-AGPL_v3-blue.svg)](https://www.gnu.org/licenses/agpl-3.0)
[![Code Style: Black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/ambv/black)
[![Code Coverage](https://codecov.io/gh/a-t-0/snn/branch/main/graph/badge.svg)](https://codecov.io/gh/a-t-0/snnalgorithms)

This module compares SNN \[algorithms\] to their default/Neumann implementations.
The user can specify an SNN and "normal" algorithm which take as input a
networkx graph, and compute some graph property as output. The output of the
SNN is then compared to the "normal" algorithm as "ground truth", in terms of:

- Score\*: How many results the SNN algorithm computed correctly (from a set of input
  graphs).
- Runtime
- Energy Complexity (nr of spikes)
- Space Complexity (nr of neurons)
- Connectivity (nr of synapses)
- Radiation Robustness

\*In theory, the score should always be 100% for the SNN, as it should be an
exact SNN implementation of the ground truth algorithm. This comparison is
mainly relevant for the additions of brain adaptation and simulated radiation.

## Example

Below is an example of the SNN behaviour of the MDSA algorithm without
adaptation, without radiation, on a (non-triangular) input graph of 5 nodes.
<img src="example.gif" width="1280" height="960" />

The green dots are when the neurons spike, non-spiking neurons are yellow.

## Brain adaptation

For each SNN algorithm that the user specifies, the user can also specify a
form of brain-inspired adaptation. This serves to increase the robustness of
the SNN against radiation effects. The \[brain-adaptation\] can be called from a
separate pip package called: `snnadaptation`.

## Radiation

A basic form of \[radiation\] effects is modelled on the SNNs. For example,
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

In hardware platforms where neurons and synapses have a more physical
implementation on chip, the adaptation may be more effective to increase the
radiation robustness.

## Backends

Since the effectiveness of the adaptation mechanisms, in terms of radiation
robustness, is a function of neuromorphic hardware platform, multiple \[backends\]
are supported. These backends also allow for different neuronal and synaptic
models. Currently the following backends are supported:

- A self-made networkx SNN simulator (LIF-neurons)
- Lava-nc simulator v0.5.0 (LIF-neurons)

## Algorithms

Different SNN implementations may use different encoding schemes, such as
sparse coding, population coding and/or rate coding. In population coding,
adaptation may be realised in the form of larger populations, whereas in rate
coding, adaptation may be realised through varying the spike-rate. This implies
that different algorithms may benefit from different types of adaptation.
Hence, an overview is included of the implemented SNN algorithms and their
respective compatibilities with adaptation and radiation implementations:

| Algorithm                            | Encoding | Adaptation | Radiation    |
| ------------------------------------ | -------- | ---------- | ------------ |
| Minimum Dominating Set Approximation | Sparse   | Redundancy | Neuron Death |
|                                      |          |            |              |
|                                      |          |            |              |

### Minimum Dominating Set Approximation

This is an implementation of the distributed algorithm presented by Alipour et al.

- *Input*: Non-triangle, planar Networkx graph. (Non triangle means there
  should not be any 3 nodes that are all connected with each other (forming a
  triangle)). Planar means that if you lay-out the graph on a piece of paper, no
  lines intersect (that you can roll it out on a 2D plane).
- *Output*: A set of nodes that form a dominating set in the graph.

*Description:* The algorithm basically consists of `k` rounds, where you can
choose `k` based on how accurate you want the approximation to be, more rounds
(generally) means more accuracy. At the start each node `i` gets 1 random
number `r_i`. This is kept constant throughout the entire algorithm. Then for
the first round:

- Each node `i` computes how many neighbours (degree) `d_i` it has.
- Then it adds `r_i+d_i=w_i`.
  In all consecutive rounds:
- Each node `i` "computes" which neighbour has the highest weight `w_j`, and
  gives that node 1 mark/point. Then each node `i` has some mark/score `m_i`.
  Next, the weight `w_i=r_i+m_i` is computed (again) and the next round starts.
  This last round is repeated until `k` rounds are completed. At the end, the
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
Stage 5: Create box plot with network performances.

## Running Experiment

First satisfy the prerequisites:

```bash
pip install snncompare
pip install https://github.com/a-t-0/lava/archive/refs/tags/v0.5.1.tar.gz
ulimit -n 800000
```

You can run the experiment (stage 1,2,4) with command:

```bash
python -m src.snncompare -e mdsa_long_no_overwrite -j1 -j2 -j4
```

This generates the graphs from the default experiment configurations, and
outputs the graphs in json format to the `results/` directory, and outputs
the graph behaviour to: `latex/Images/graphs/`.

## Additional Options

You can run the experiment (stage 1,2,4) in reverse (from small to large
graphs) with command:

```bash
python -m src.snncompare -e mdsa_long_no_overwrite -j1 -j2 -j4 -rev
```

You can run a single `run_config` with:

```bash
python -m src.snncompare -e mdsa_long_no_overwrite -j1 -j2 -j4 -r run_config_file_name
```

Typical run (deletes pre-existing results):

```bash
python -m src.snncompare -e quicktest -j1 -j2 -j4 -j5 -s2 -rev -dr
python -m src.snncompare -e basic_results -j1 -j2 -j4 -j5 -s2 -rev -dr
```

Debug 2 runs, in separate console:

```bash
python -m src.snncompare -e debug0 -j1 -j2 -j4 -j5 -rev  -si -sgt \
 rad_adapted_snn_graph -p 8000
python -m src.snncompare -e debug1 -j1 -j2 -j4 -j5 -rev  -si -sgt \
 rad_adapted_snn_graph -p 8003
```

For more info, run:

```bash
python -m src.snncompare --help
```

And run tests with:

```bash
python -m pytest
```

or to see live output, on any tests filenames containing substring: `results`:

```bash
python -m pytest tests/sparse/MDSA/test_snn_results_with_adaptation.py --capture=tee-sys
```

## Developers

Improve the project using:

```bash
mkdir -p ~/git/snn
mkdir ~/git/snn/.vscode
mkdir -p ~/bin
cd ~/git/snn

git clone https://github.com/a-t-0/snnadaptation.git
git clone https://github.com/a-t-0/snnalgorithms.git
git clone https://github.com/a-t-0/snnbackends.git
git clone https://github.com/a-t-0/snnradiation.git
git clone https://github.com/a-t-0/snncompare.git
git clone https://gitlab.socsci.ru.nl/Akke.Toeter/simsnn.git

cd snncompare
conda env create --file environment.yml
git checkout excitatory-radiation
chmod +x snnrb
./snnrb --branch excitatory-radiation
./snnrb --rebuild

cp snncompare/.vscode/settings.json .vscode/settings.json
```

Then you can commit/update your work across all repos at  once with:

```bash
snnrb -c "Some commit."
```

## Test Coverage

Developers can use:

```bash
conda env create --file environment.yml
conda activate snncompare
ulimit -n 800000
python -m pytest
```

Currently the test coverage is `65%`. For type checking:

```bash
mypy --disallow-untyped-calls --disallow-untyped-defs tests/export_results/performed_stage/test_performed_stage_TTFF.py
```

### Releasing pip package update

To udate the Python pip package, one can first satisfy the following requirements:

```bash
pip install --upgrade pip setuptools wheel
pip install twine
```

Followed by updating the package with:

```bash
python3 setup.py sdist bdist_wheel
python -m twine upload dist/\*
```

### Developer pip install

```bash
mkdir -p ~/bin
cp snn_rebuild.sh ~/.local/bin/snnrb
chmod +x ~/bin/snnrb
```

### Updating

Build the pip package with:

```bash
pip install --upgrade pip setuptools wheel
pip install twine
```

Install the pip package locally with:

```bash
rm -r dist
rm -r build
python3 setup.py sdist bdist_wheel
pip install -e .
```

Upload the pip package to the world with:

```bash
rm -r dist
rm -r build
python3 setup.py sdist bdist_wheel
python3 -m twine upload dist/\*
```
