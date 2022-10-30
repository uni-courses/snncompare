"""Parses CLI arguments that specify on which platform to simulate the spiking
neural network (SNN)."""
import argparse


def parse_cli_args() -> argparse.Namespace:
    """Reads command line arguments and converts them into python arguments."""

    # Instantiate the parser
    parser = argparse.ArgumentParser(
        description="Optional description for arg" + " parser"
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

    # Run experiment on a particular experiment_settings json file.
    parser.add_argument(
        "-e",
        "--experiment-settings-path",
        action="store",
        type=str,
        help=(
            "Give filepath to experiment settings json on which to run "
            + "the experiment."
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

    # Ensure SNN behaviour is visualised in stage 3.
    parser.add_argument(
        "-v",
        "--visualise-snn",
        action="store_true",
        default=False,
        help=("Ensures the SNN behaviour is visualised."),
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
            + "implies later stages are rewritten as well, images deleted, "
            + "unless specified otherwise unless specified otherwise (with: "
            + "--keep-...)"
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
            "already existed. implies later stages are rewritten as well, "
            "images deleted, unless specified otherwise (with: --keep-...)"
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
            "already existed."
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
            "already existed."
        ),
    )

    # Create argument parsers to allow user to overwrite pre-existing output.

    # Create argument parsers to allow user specify experiment config in CLI.
    # TODO

    # Load the arguments that are given.
    args = parser.parse_args()
    return args
