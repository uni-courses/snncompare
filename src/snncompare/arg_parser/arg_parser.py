"""Parses CLI arguments that specify on which platform to simulate the spiking
neural network (SNN)."""
import argparse

from typeguard import typechecked

from snncompare.exp_config.Exp_config import Supported_experiment_settings


@typechecked
def parse_cli_args() -> argparse.Namespace:
    """Reads command line arguments and converts them into python arguments."""
    supp_setts = Supported_experiment_settings()

    # Instantiate the parser
    parser = argparse.ArgumentParser(
        description="Optional description for arg" + " parser"
    )

    parser.add_argument(
        "-c",
        "--create_boxplots",
        action="store_true",
        default=False,
        help=("Create boxplots with adaptation effectivity."),
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

    # Allow user to set graph size.
    parser.add_argument(
        "-m",
        "--m_val",
        nargs="?",
        type=int,
        dest="m_val",
        const="m_val",
        help=("Specify the m_val on which to run the MDSA algorithm."),
    )

    # Create argument parsers to allow user to overwrite pre-existing output.
    # Ensure new SNN graphs are created.
    parser.add_argument(
        "-oc",
        "--overwrite-creation",
        action="store_true",
        default=False,
        help=(
            "Ensures new SNN graph is created, even if it already existed."
            + "use this to only overwrite specific runs without deleting the"
            + "entire results jsons."
        ),
    )

    # Ensure new SNN graph propagation is performed.
    parser.add_argument(
        "-op",
        "--overwrite-propagation",
        action="store_true",
        default=False,
        help=(
            "Ensures new SNN graph propagation is performed, even if it "
            + "already existed. Use this to only overwrite specific runs "
            + "without deleting the entire results jsons."
        ),
    )

    # Ensure new SNN graph propagation is performed.
    parser.add_argument(
        "-or",
        "--overwrite-results",
        action="store_true",
        default=False,
        help=(
            "Ensures new SNN algorithm results are computed, even if they "
            + "already existed. Use this to only overwrite specific runs "
            + "without deleting the entire results jsons."
        ),
    )

    # Ensure new SNN graph behaviour visualistation is created.
    parser.add_argument(
        "-ov",
        "--overwrite-visualisation",
        action="store_true",
        default=False,
        help=(
            "Ensures new SNN graph behaviour is visualised, even if it "
            + "already existed. Use this to only overwrite specific runs "
            + "without deleting the entire image section."
        ),
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

    # Allow user to set a neuron redundancy value.
    parser.add_argument(
        "-rd",
        "--redundancy",
        nargs="?",
        type=int,
        dest="redundancy",
        const="redundancy",
        help=("Specify the redundancy used as adaptation mechanism."),
    )

    # Allow user to set graph size.
    parser.add_argument(
        "-s",
        "--graph-size",
        nargs="?",
        type=int,
        dest="graph_size",
        const="graph_size",
        help=(
            "Specify the graph size on which to run algorithm. Performs a "
            "single run by default. Assume you want to run a single iteration."
        ),
    )

    # Ensure SNN behaviour is visualised in stage 3.
    parser.add_argument(
        "-v",
        "--visualise-snn",
        action="store_true",
        default=False,
        help=("Pause computation, show you each plot of the SNN behaviour."),
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
            + f"{supp_setts.export_types.extend(['gif','zoom'])}. Usage:"
            + f'-x {",".join(supp_setts.export_types+["gif","zoom"])} '
            + "or:\n"
            + f"--export_images {supp_setts.export_types[0]}"
        ),
    )

    # Create argument parsers to allow user to overwrite pre-existing output.

    # Create argument parsers to allow user specify experiment config in CLI.
    # TODO

    # Load the arguments that are given.
    args = parser.parse_args()
    return args
