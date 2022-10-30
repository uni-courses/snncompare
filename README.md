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


============================ test session starts ==============================
platform linux -- Python 3.8.13, pytest-7.2.0, pluggy-1.0.0
rootdir: /home/name/git/snncompare, configfile: pyproject.toml, testpaths: tests
plugins: cov-4.0.0
collected 82 items                                                                                                                           

tests/experiment_settings/test_generic_experiment_settings.py ....               [  4%]
tests/experiment_settings/test_generic_run_settings.py ....                      [  9%]
tests/experiment_settings/test_supported_settings_adaptation.py ......           [ 17%]
tests/experiment_settings/test_supported_settings_iteration.py .....             [ 23%]
tests/experiment_settings/test_supported_settings_m_vals.py .....                [ 29%]
tests/experiment_settings/test_supported_settings_max_graph_size.py .....        [ 35%]
tests/experiment_settings/test_supported_settings_max_max_graphs.py .....        [ 41%]
tests/experiment_settings/test_supported_settings_min_graph_size.py ....         [ 46%]
tests/experiment_settings/test_supported_settings_min_max_graphs.py ....         [ 51%]
tests/experiment_settings/test_supported_settings_overwrite_sim_results.py ..    [ 53%]
tests/experiment_settings/test_supported_settings_overwrite_visualisation.py ..  [ 56%]
tests/experiment_settings/test_supported_settings_radiation.py ......            [ 63%]
tests/experiment_settings/test_supported_settings_seed.py ..                     [ 65%]
tests/experiment_settings/test_supported_settings_simulators.py ....             [ 70%]
tests/experiment_settings/test_supported_settings_size_and_max_graphs.py .....   [ 76%]
tests/export_results/performed_stage/test_performed_stage_TFFF.py .              [ 78%]
tests/export_results/performed_stage/test_performed_stage_TTFF.py .              [ 79%]
tests/export_results/performed_stage/test_performed_stage_TTTF.py .              [ 80%]
tests/export_results/performed_stage/test_performed_stage_TTTT.py .              [ 81%]
tests/graph_generation/test_get_graph.py .                                       [ 82%]
tests/graph_generation/test_rand_network_generation.py ..                        [ 85%]
tests/simulation/test_cyclic_graph_propagation.py .                              [ 86%]
tests/simulation/test_has_recurrent_edge_lava.py .....                           [ 92%]
tests/simulation/test_has_recurrent_edge_networkx.py .....                       [ 98%]
tests/simulation/test_rand_network_propagation.py .                              [100%]


---------- coverage: platform linux, python 3.8.13-final-0 -----------
Name                                                            Stmts   Miss  Cover
-----------------------------------------------------------------------------------
src/__init__.py                                                     0      0   100%
src/snncompare/__init__.py                                          2      0   100%
src/snncompare/__main__.py                                         13     13     0%
src/snncompare/exp_setts/Adaptation_Rad_settings.py                16      0   100%
src/snncompare/exp_setts/Experiment_runner.py                     110      9    92%
src/snncompare/exp_setts/Scope_of_tests.py                         39     17    56%
src/snncompare/exp_setts/Supported_algorithms.py                    5      0   100%
src/snncompare/exp_setts/Supported_experiment_settings.py          57      4    93%
src/snncompare/exp_setts/Supported_run_settings.py                 18      2    89%
src/snncompare/exp_setts/__init__.py                                0      0   100%
src/snncompare/exp_setts/create_testobject.py                      61     45    26%
src/snncompare/exp_setts/verify_experiment_settings.py            123     12    90%
src/snncompare/exp_setts/verify_run_completion.py                   4      1    75%
src/snncompare/exp_setts/verify_run_settings.py                    31      8    74%
src/snncompare/export_results/Output.py                           114     80    30%
src/snncompare/export_results/Output_stage_12.py                    5      0   100%
src/snncompare/export_results/Output_stage_34.py                   13      5    62%
src/snncompare/export_results/Plot_to_tex.py                       75     59    21%
src/snncompare/export_results/__init__.py                           0      0   100%
src/snncompare/export_results/check_json_graphs.py                 12     12     0%
src/snncompare/export_results/check_nx_graphs.py                    9      7    22%
src/snncompare/export_results/export_json_results.py               37     21    43%
src/snncompare/export_results/export_nx_graph_to_json.py           42      8    81%
src/snncompare/export_results/helper.py                            39      2    95%
src/snncompare/export_results/load_json_to_nx_graph.py             64     31    52%
src/snncompare/export_results/load_pickles_get_results.py          22     18    18%
src/snncompare/export_results/plot_graphs.py                       36      9    75%
src/snncompare/export_results/verify_json_graphs.py                21      5    76%
src/snncompare/export_results/verify_nx_graphs.py                  42     10    76%
src/snncompare/export_results/verify_stage_1_graphs.py             23     10    57%
src/snncompare/export_results/verify_stage_2_graphs.py              1      0   100%
src/snncompare/export_results/verify_stage_3_graphs.py              1      0   100%
src/snncompare/export_results/verify_stage_4_graphs.py              1      0   100%
src/snncompare/graph_generation/Used_graphs.py                     95     25    74%
src/snncompare/graph_generation/__init__.py                         0      0   100%
src/snncompare/graph_generation/adaptation/__init__.py              0      0   100%
src/snncompare/graph_generation/adaptation/redundancy.py           59      5    92%
src/snncompare/graph_generation/brain_adaptation.py                78     78     0%
src/snncompare/graph_generation/convert_networkx_to_lava.py       117      9    92%
src/snncompare/graph_generation/get_graph.py                       76      3    96%
src/snncompare/graph_generation/helper_network_structure.py       144    114    21%
src/snncompare/graph_generation/radiation/Radiation_damage.py      71     28    61%
src/snncompare/graph_generation/radiation/__init__.py               0      0   100%
src/snncompare/graph_generation/snn_algo/__init__.py                0      0   100%
src/snncompare/graph_generation/snn_algo/mdsa_snn_algo.py         110      1    99%
src/snncompare/graph_generation/stage_1_get_input_graphs.py        85     14    84%
src/snncompare/helper.py                                          276    167    39%
src/snncompare/import_results/__init__.py                           0      0   100%
src/snncompare/import_results/check_completed_stages.py            77     21    73%
src/snncompare/import_results/read_json.py                         37      4    89%
src/snncompare/import_results/stage_1_load_input_graphs.py         28     20    29%
src/snncompare/old_conversion.py                                   59     49    17%
src/snncompare/process_results/__init__.py                          0      0   100%
src/snncompare/process_results/get_alipour_nodes.py                17      0   100%
src/snncompare/process_results/get_mdsa_results.py                 54      3    94%
src/snncompare/process_results/process_results.py                  36      3    92%
src/snncompare/process_results/stage_2_propagate_graphs.py          0      0   100%
src/snncompare/simulation/LIF_neuron.py                           120     11    91%
src/snncompare/simulation/__init__.py                               0      0   100%
src/snncompare/simulation/run_on_lava.py                           24      2    92%
src/snncompare/simulation/run_on_networkx.py                       51      3    94%
src/snncompare/simulation/stage2_sim.py                            25      3    88%
src/snncompare/simulation/verify_graph_is_lava_snn.py              14      4    71%
src/snncompare/simulation/verify_graph_is_networkx_snn.py          31      7    77%
src/snncompare/simulation/verify_graph_is_snn.py                   12      0   100%
src/snncompare/verification_generic.py                             10      2    80%
-----------------------------------------------------------------------------------
TOTAL                                                            2742    964    65%
Coverage XML written to file cov.xml


=================== 82 passed, 1 warning in 415.57s (0:06:55) =======================

```
