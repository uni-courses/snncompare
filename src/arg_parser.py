# Parses CLI arguments that specify on which platform to simulate the spiking
# neural network (SNN).
import argparse


def parse_cli_args() -> argparse.Namespace:
    """Reads command line arguments and converts them into python arguments."""
    # Instantiate the parser
    parser = argparse.ArgumentParser(
        description="Optional description for arg" + " parser"
    )

    # Include argument parsing for default code.
    # Allow user to load a graph from file.
    parser.add_argument(
        "--g",
        dest="graph_from_file",
        action="store_true",
        help="boolean flag, determines whether graph is created from file.",
    )

    # Allow user to specify an infile.
    parser.add_argument("infile", nargs="?", type=argparse.FileType("r"))

    parser.add_argument(
        "--nx",
        dest="run_on_networkx",
        action="store_true",
        help="boolean flag, determines whether snn is simulated using"
        + " networkx instead of Lava.",
    )

    parser.add_argument(
        "--l",
        dest="run_on_lava",
        action="store_true",
        help="boolean flag, determines whether snn is simulated using"
        + " lava instead of networkx.",
    )

    # Specify default argument values for the parser.
    parser.set_defaults(
        infile=None,
        graph_from_file=False,
    )

    # Load the arguments that are given.
    args = parser.parse_args()
    return args
