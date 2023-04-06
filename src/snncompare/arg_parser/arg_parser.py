"""Parses CLI arguments that specify on which platform to simulate the spiking
neural network (SNN)."""
from argparse import ArgumentParser, Namespace
from typing import Optional, Union

from typeguard import typechecked

from snncompare.exp_config.Exp_config import Supported_experiment_settings


@typechecked
def parse_cli_args(
    parse: Optional[bool] = True,
) -> Union[ArgumentParser, Namespace]:
    """Reads command line arguments and converts them into python arguments."""
    supp_setts = Supported_experiment_settings()

    # Instantiate the parser
    parser = ArgumentParser(
        description="Optional description for arg" + " parser"
    )

    parser.add_argument(
        "-c",
        "--create-boxplots",
        action="store_true",
        default=False,
        help=(""),
    )

    parser.add_argument(
        "-cf",
        "--count-fires",
        action="store_true",
        default=False,
        help=(
            'Store how many "spikes=neuron firing events" were registered in '
            + "each graph propagation."
        ),
    )

    parser.add_argument(
        "-cs",
        "--count-synapses",
        action="store_true",
        default=False,
        help=("Store how many synapses were created in each SNN."),
    )

    parser.add_argument(
        "-cn",
        "--count-neurons",
        action="store_true",
        default=False,
        help=("Store how many neurons were created in each SNN."),
    )

    parser.add_argument(
        "-di",
        "--delete-images",
        action="store_true",
        default=False,
        help=(
            "Delete the images in the /latex/graphs/ directory at the start "
            + "of the experiment."
        ),
    )

    parser.add_argument(
        "-dr",
        "--delete-results",
        action="store_true",
        default=False,
        help=(
            "Delete the snn graphs, propagation, and results in the results/"
            + " dir."
        ),
    )

    # Run experiment on a particular experiment_settings json file.
    parser.add_argument(
        "-e",
        "--experiment-settings-name",
        action="store",
        default="mdsa_size3_m0",  # Load default/minimal experiment settings.
        type=str,
        help=(
            "Give filename to experiment settings json on which to run "
            + "the experiment."
        ),
    )

    # Create argument parsers to allow user to specify what to run.
    # Allow user run the experiment on a graph from file.
    parser.add_argument(
        "-g",
        "--graph-filepath",
        action="store",
        type=str,
        help=(
            "Run default experiment on networkx graph in json filepath. Give "
            + "the filepath."
        ),
    )

    # Run run on a particular run_settings json file.
    parser.add_argument(
        "-j1",
        "--output-json-stage-1",
        action="store_true",
        default=False,
        help=(
            "Store the json output for stage 1: initialised graphs that are "
            + "to be propagated in stage 2."
        ),
    )

    # Run run on a particular run_settings json file.
    parser.add_argument(
        "-j2",
        "--output-json-stage-2",
        action="store_true",
        default=False,
        help=(
            "Store the json output for stage 2: networkx graphs that are "
            + "propagated in stage 2."
        ),
    )

    parser.add_argument(
        "-j4",
        "--output-json-stage-4",
        action="store_true",
        default=False,
        help=(
            "Store the json output for stage 4: the results of the SNN graphs."
        ),
    )

    parser.add_argument(
        "-j5",
        "--output-json-stage-5",
        action="store_true",
        default=False,
        help=("Store a compact json output with the boxplot data."),
    )

    # Run run on a particular run_settings json file.
    parser.add_argument(
        "-r",
        "--run-config-path",
        action="store",
        type=str,
        help=(
            "Give filepath to run settings json on which to run " + "the run."
        ),
    )

    parser.add_argument(
        "-r1",
        "--recreate-stage-1",
        action="store_true",
        default=False,
        help=(
            "Ensures new SNN graph is created, even if it already existed."
            + "use this to only overwrite specific runs without deleting the"
            + "entire results jsons."
        ),
    )

    # Run run on a particular run_settings json file.
    parser.add_argument(
        "-r2",
        "--recreate-stage-2",
        action="store_true",
        default=False,
        help=(
            "Ensures new SNN graph propagation is performed, even if it "
            + "already existed. Use this to only overwrite specific runs "
            + "without deleting the entire results jsons."
        ),
    )

    # Run run on a particular run_settings json file.
    parser.add_argument(
        "-r3",
        "--recreate-stage-3",
        action="store_true",
        default=False,
        help=(
            "Ensures new SNN graph behaviour is visualised, even if it "
            + "already existed. Use this to only overwrite specific runs "
            + "without deleting the entire image section. You also have to"
            + " specify the image export types with: -x png,gif,svg etc."
        ),
    )

    # Run run on a particular run_settings json file.
    parser.add_argument(
        "-r4",
        "--recreate-stage-4",
        action="store_true",
        default=False,
        help=(
            "Ensures new SNN algorithm results are computed, even if they "
            + "already existed. Use this to only overwrite specific runs "
            + "without deleting the entire results jsons."
        ),
    )

    parser.add_argument(
        "-r5",
        "--recreate-stage-5",
        action="store_true",
        default=False,
        help=("Rereate boxplots with adaptation effectivity."),
    )

    # Run run on a particular run_settings json file.
    parser.add_argument(
        "-rev",
        "--reverse",
        action="store_true",
        default=False,
        help=("Run experiment config from small/fast to large/slow."),
    )

    parser.add_argument(
        "-si",
        "--show-images",
        action="store_true",
        default=False,
        help=(
            "Show images in dash app/browser. (Automatically sets export"
            + "images to svg)."
        ),
    )

    parser.add_argument(
        "-sgt",
        "--show-graph-type",
        type=str,
        dest="show_graph_type",
        # const="show_graph_type",
        help=(
            "If show-images is true, this will allow you to specify which"
            + "graph type  is shown in Dash. You can choose from:"
            + " - rad_adapted_snn_graph"
            + " - adapted_snn_graph"
            + " - rad_snn_algo_graph"
            + " - snn_algo_graph"
        ),
    )

    parser.add_argument(
        "-sdn",
        "--store-died-neurons",
        action="store_true",
        default=False,
        help=("Store which neurons died due to radiation."),
    )

    # Ensure SNN behaviour visualisation in stage 3 is exported to images.
    parser.add_argument(
        "-x",
        "--export-images",
        nargs="?",
        type=str,
        dest="export_images",
        const="export_images",
        help=(
            "Ensures the SNN behaviour visualisation is exported, as pdf by "
            + "default. Supported are:"
            + f"{supp_setts.export_types.extend(['gif'])}. Usage:"
            + f'-x {",".join(supp_setts.export_types+["gif"])} '
            + "or:\n"
            + f"--export_images {supp_setts.export_types[0]}"
        ),
    )

    # Ensure SNN behaviour visualisation in stage 3 is exported to images.
    parser.add_argument(
        "-p",
        "--port",
        nargs="?",
        type=int,
        dest="dash_port",
        help=("Show dash app in browser on 127:0.0.1:<port>"),
    )

    # Ensure SNN behaviour visualisation in stage 3 is exported to images.
    parser.add_argument(
        "-z",
        "--zoom",
        nargs="?",
        type=str,
        dest="zoom",
        const="zoom",
        help=(
            "Create zoomed images for output png files of SNN behaviour. "
            + "Give the left, right, bottom and top fraction of the image you "
            + "want to see like: -z 0.1,0.5,0.7,0.9 to see the top left part.:"
        ),
    )

    # Load the arguments that are given and execute them.
    if parse:
        args = parser.parse_args()
        return args
    return parser
