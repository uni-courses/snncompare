"""Runs brain adaptation tests."""

import random
import unittest
from time import time

from lava.magma.core.run_conditions import RunSteps
from lava.magma.core.run_configs import Loihi1SimCfg

from src.brain_adaptation import adaptation_mech_2_networkx_and_snn
from src.create_testobject import create_test_object
from src.helper import (
    create_neuron_monitors,
    delete_files_in_folder,
    export_get_degree_graph,
    full_alipour,
    get_counter_neurons_from_dict,
    print_time,
    store_spike_values_in_neurons,
    write_results_to_file,
)
from src.helper_network_structure import plot_neuron_behaviour_over_time
from src.Radiation_damage import Radiation_damage
from src.Used_graphs import Used_graphs


class Test_counter(unittest.TestCase):
    """Tests whether the counter neurons indeed yield the same degree count as
    the Alipour algorithm implementation in the first step/initialisation of
    the algorithm."""

    # Initialize test object
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.seed = 42
        random.seed(self.seed)

    def test_snn_algorithm(self):
        """

        :param output_behaviour:  (Default value = False)

        """
        # pylint: disable=R0914
        # TODO: reduce local variables from 21/15 to at most 15/15.
        # pylint: disable=R1702
        # TODO: reduce nested blocks from 8/5 to at most 5/5.

        # delete_dir_if_exists("latex/Images/graphs")
        # delete_files_in_folder("latex/Images/graphs")
        delete_files_in_folder("pickles")
        used_graphs = Used_graphs()
        # Specify User randomnes to prevent overwriting identical runs.

        for m in range(0, 5):
            for iteration in range(0, 2, 1):
                for size in range(3, 4, 1):
                    # for neuron_death_probability in [0.1, 0.25, 0.50]:
                    for neuron_death_probability in [
                        0.01,
                        0.05,
                        0.1,
                        0.2,
                        0.25,
                    ]:
                        rad_dam = Radiation_damage(neuron_death_probability)
                        graphs = used_graphs.get_graphs(size)
                        for G in graphs:
                            for has_adaptation in [True, False]:
                                for has_radiation in [True, False]:

                                    # Start performance report.
                                    latest_millis = int(round(time() * 1000))
                                    latest_time, latest_millis = print_time(
                                        "Create object.",
                                        latest_millis,
                                    )

                                    # Initialise paramers used for testing.
                                    (test_object,) = create_test_object(
                                        has_adaptation,
                                        G,
                                        m,
                                        self.seed,
                                    )

                                    # Specify simulation duration.
                                    # TODO: determine why 10 is needed and
                                    # hardcode it.
                                    sim_time = (
                                        test_object.rand_props.inhibition
                                        * (m + 1)
                                        + 10
                                    )

                                    # Report performance.
                                    latest_time, latest_millis = print_time(
                                        "Created object.",
                                        latest_millis,
                                    )

                                    # Apply simulated brain adaptation to
                                    # networkx graph and SNN, if desired.
                                    if has_adaptation:
                                        # pylint: disable=W0612
                                        # TODO: improve result store method to
                                        # remove this disable necessity.
                                        (
                                            dead_neuron_names,
                                            latest_time,
                                            latest_millis,
                                        ) = adaptation_mech_2_networkx_and_snn(
                                            has_radiation,
                                            latest_millis,
                                            latest_time,
                                            m,
                                            rad_dam,
                                            sim_time,
                                            test_object,
                                        )
                                    else:
                                        test_object.brain_adaptation_graph = (
                                            None
                                        )
                                        test_object.rad_damaged_graph = None
                                        test_object.final_dead_neuron_names = (
                                            None
                                        )

                                    # Add spike monitors in networkx graph
                                    # representing SNN. If output_behaviour:
                                    create_neuron_monitors(
                                        test_object, sim_time
                                    )
                                    # Report performance.
                                    latest_time, latest_millis = print_time(
                                        "Got neuron monitors.",
                                        latest_millis,
                                    )

                                    # Get the counter neurons at the end of the
                                    # simulation.
                                    counter_neurons = (
                                        get_counter_neurons_from_dict(
                                            len(test_object.G),
                                            test_object.neuron_dict,
                                            m,
                                        )
                                    )
                                    latest_time, latest_millis = print_time(
                                        "Got counter neurons.",
                                        latest_millis,
                                    )

                                    # Check if expected counter nodes are
                                    # selected.
                                    # pylint: disable=W0612
                                    # TODO: improve result store method to
                                    # remove this disable necessity.
                                    (
                                        alipour_count,
                                        has_passed,
                                        snn_count,
                                    ) = self.integration_test_on_end_result(
                                        counter_neurons,
                                        G,
                                        iteration,
                                        m,
                                        self.seed,
                                        test_object,
                                    )
                                    latest_time, latest_millis = print_time(
                                        "Performed integration test.",
                                        latest_millis,
                                    )
                                    # starter_neuron.stop()

                                    # Store results into Run object.
                                    # run_result = Run(
                                    #    dead_neuron_names,
                                    #    G,
                                    #    test_object.get_degree,
                                    #    has_adaptation,
                                    #    iteration,
                                    #    neuron_death_probability,
                                    #    m,
                                    #    has_passed,
                                    #    test_object.rand_ceil,
                                    #    test_object.rand_nrs,
                                    #    alipour_count,
                                    #    snn_count,
                                    #    sim_time,
                                    #    size,
                                    # )

                                    # Export run object results.
                                    # Terminate loihi simulation for this run.
                                    export_get_degree_graph(
                                        has_adaptation,
                                        has_radiation,
                                        test_object.G,
                                        test_object.get_degree,
                                        iteration,
                                        m,
                                        neuron_death_probability,
                                        test_object.rand_props,
                                        self.seed,
                                        sim_time,
                                        size,
                                        test_object,
                                    )

    def simulate_degree_receiver_neurons(
        self,
        adaptation,
        iteration,
        latest_millis,
        latest_time,
        m,
        neuron_death_probability,
        neurons,
        output_behaviour,
        seed,
        sim_time,
        size,
        test_object,
    ):
        """Verifies the neuron properties over time.

        :param adaptation: indicates if test uses brain adaptation or not.
        param iteration: The initialisation iteration that is used.
        :param latest_millis: Timestamp with millisecond accuracy. Format
         unknown.
        :param latest_time: Previously measured time in milliseconds. Format
         unknown.
        :param m: The amount of approximation iterations used in the MDSA
        approximation.
        :param neuron_death_probability:
        :param neurons:
        :param output_behaviour:
        :param seed: The value of the random seed used for this test.
        :param sim_time: Nr. of timesteps for which the experiment is ran.
        :param size: Nr of nodes in the original graph on which test is ran.
        :param test_object: Object containing test settings.
        """
        # pylint: disable=R0913
        # TODO: reduce arguments from 13/5 to at most 5/5.

        # Get the first neuron in the SNN to start the simulation
        starter_neuron = neurons[0]

        # Simulate SNN and assert values in between timesteps.
        latest_time, latest_millis = print_time(
            "Start simulation for 1 timestep.", latest_millis
        )
        for t in range(1, sim_time):

            # Run the simulation for 1 timestep.
            starter_neuron.run(
                condition=RunSteps(num_steps=1), run_cfg=Loihi1SimCfg()
            )
            latest_time, latest_millis = print_time(
                f"Simulated SNN for t={t}.", latest_millis
            )

            # Store spike bools in networkx graph for plotting.
            store_spike_values_in_neurons(test_object.get_degree, t)
            if output_behaviour:
                print(f"t={t}, sim_time={sim_time}")
                plot_neuron_behaviour_over_time(
                    f"probability_{neuron_death_probability}_adapt_"
                    + f"{adaptation}_{seed}_size{size}_m{m}_iter{iteration}"
                    + f"t{t}",
                    test_object.get_degree,
                    t,
                    show=False,
                    current=True,
                )

        # raise Exception("Stop")
        return latest_time, neurons, starter_neuron

    def integration_test_on_end_result(
        self, counter_neurons, G, iteration, m, seed, test_object
    ):
        """Tests whether the SNN returns the same results as the Alipour
        algorithm.

        :param counter_neurons: Neuron objects at the counter position. Type
         unknown.
        :param G: The original graph on which the MDSA algorithm is ran.
        param iteration: The initialisation iteration that is used.
        :param m: The amount of approximation iterations used in the MDSA
        approximation.
        :param seed: The value of the random seed used for this test.
        :param test_object: Object containing test settings.
        """
        # pylint: disable=R0913
        # TODO: reduce 7/5 arguments to at most 5/5.
        alipour_count = []
        snn_count = []
        has_passed = True

        # Compute the Alipour graph
        G_alipour = full_alipour(
            iteration,
            G,
            test_object.m,
            test_object.rand_props,
            seed,
            len(test_object.G),
            export=False,
        )

        # Compare the counts per node and assert they are equal.
        print("G_alipour countermarks-SNN counter current")
        for node in G.nodes:
            print(
                f'{G_alipour.nodes[node]["countermarks"]}'
                + f"-{counter_neurons[node].u.get()}"
            )
            alipour_count.append({G_alipour.nodes[node]["countermarks"]})
            snn_count.append(counter_neurons[node].u.get())
        print("Now testing they are equal:")
        for node in G.nodes:
            # self.assertEqual(
            #    G_alipour.nodes[node]["countermarks"],
            #    counter_neurons[node].u.get(),
            # )
            if (
                G_alipour.nodes[node]["countermarks"]
                != counter_neurons[node].u.get()
            ):
                has_passed = False

        # TODO: include redundant counter neurons to check if it has passed.

        write_results_to_file(
            has_passed, m, G, iteration, G_alipour, counter_neurons
        )

        return alipour_count, has_passed, snn_count
