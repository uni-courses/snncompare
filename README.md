# networkx-to-lava-nc

This intermediate module is used to improve the simulation speed and stability
of spiking neural networks that consist of leaky-integrate-and-fire (LIF)
neurons of the LAVA software by Intel. These lava LIF neurons are simulated and
ran on the networkx graph module to improve speed and stability. It is
compatible with Lava-NC, hence one can choose on which platform one wants to
simulate the SNN (networkx or Lava). The improved speed and stability allow
for an increase in SNN size and simulation time. Hence, the brain-adaptation
implementation on the SNNs and accompanying radiation exposure simulation are
ran on the networkx-module to generate more robust test results. The improved
stability allows for a larger Minimum DominatingSet Approximation instance to
be tested, which in turn allows for more graphs to be used in the simulated
radiation robustness tests. The high level description of this intermediate
module is given by:

- *Input*: Networkx graphs specifying a spiking neural network (SNN).
- *Output*: Runnable Lava SNNs consisting of LIF neurons and synapses.

## Running Experiment

Temporary instructions, will be updated after refactoring.
First generate the `.pickle` files with:

1. The original graph on which the algorithm is ran.
1. The MDSA SNN networkx graph.
1. The MDSA SNN networkx graph with adaptation mechanism.
1. The MDSA SNN networkx graph with adaptation mechanism and radiation.
   depending on the experiment configuration.

### Experiment Stages

The experiment is ran in multiple stages, these are:
Input: Experiment configuration. Which consists of:
SubInput: Run configuration within an experiment.
Stage 1: Create networkx graphs that will be propagated.
Stage 2: Create propagated networkx graphs (at least one per timestep).
Stage 3: Visaualisation of the networkx graphs over time.
Stage 4: Post-processed performance data of algorithm and adaptation
mechanism.

You can run the experiment with command (to run the experiment using the
networkx backend):

```
python -m src --nx
```

And run tests with:

```
python -m pytest --capture=tee-sys tests/test_get_results.py
```

Then you can run the experiment by simulating the respective graphs.
This can be done with command:

```
python -m src --pkl
```

This should simulate the graphs that are loaded from the `.pkl` file(s) in
`/pickles/` and export the experiment results to `/results/...json`.

## Usage

First set up the conda environment (see Test Coverage), then one can run:

```
python -m src --l
```

To run on the SNN simulations on the lava-nc platform by Intel, and one can run:

```
python -m src -nx
```

To run the SNN simulation on the networkx module.

## Test Coverage

The tests can be ran with:

```

conda env create --file environment.yml
conda activate nx2lava
ulimit -n
python -m pytest
```

And the respective output is listed below, with a test coverage of `85%`.:

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
