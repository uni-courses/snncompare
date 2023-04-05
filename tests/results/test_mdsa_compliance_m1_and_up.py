""""Runs the snns on a specific run config, and verifies that the degree
receiver nodes at m>=1 have the same:

- random weights
- marks
- countermarks
as the corresponding neighbours in the Neumann algorithm.
"""

# Run a run config.

# For SNN:
# Get the random values per node in that offset of the rand spike input.
# get_rand__degree_receiver_edge_weights


# Get the marks: if degree_receiver_<node_circuit>_<neighbour>_<m-1> fires/
# wins, it means it has 1 point/countermark for the node at <neighbour>,
# retrieved from the node at <node_circuit>. So in the next round, m, the
# <node_circuit> looks around again, and sees the
# degree_receiver_<node_circuit>_<neighbour>_<m>
# of <neighbour> has +1 point. Same goes for that <neighbour> in all
# other node circuits.
# So the marks are the sum of the
# degree_receiver_<node_circuit>_<neighbour>_<m-1> inputs + rand nrs.

# The weights are the values of the spike_once inputs at t=0.

# The countermarks are the nr of spike_once inputs


# For Neumann
# Get the random values per node.
# Get the marks per node
# Get the weight (=degree for m=0)
# Get the countermarks per node


# Assert the mark of Neumann equals ==
# sum of input spike_once_neurons + rand nr of that node.
# Optionally, assert that the degree_receiver - neumann mark all yield the
# same starting baseline.

# Assert Neumann weights equal spike_once input sum.
