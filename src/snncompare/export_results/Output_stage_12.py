""""Outputs the results for stage 1 and/or 2."""
from typing import Dict

from typeguard import typechecked

from .helper import run_config_to_filename
from .Output import output_stage_json


@typechecked
def output_files_stage_1_and_2(
    *, results_nx_graphs: Dict, stage_index: int
) -> None:
    """Converts the graphs of the incoming results dict into json dict graphs,

    , replaces the incoming graphs with the json dict graphs, and then exports
    the results json.

    The unique_id of the experiment is added to the file as a filetag,
    as well as the unique_id of the run. Furthermore, all run parameter
    values are added as file tags, to make it easier to filter certain
    runs to manually inspect the results.
    TODO: separate into separate method for stage 1 and for stage 2.

    :param results_nx_graphs:
    """

    # Get the json output filename.
    json_filename = run_config_to_filename(
        run_config_dict=results_nx_graphs["run_config"].__dict__
    )
    output_stage_json(
        results_nx_graphs=results_nx_graphs,
        run_config_filename=json_filename,
        stage_index=stage_index,
    )
    # Verifies while loading that the output dict contains the expected
    # completed stages.
    # load_json_to_nx_graph_from_file(results_nx_graphs["run_config"], 1)
