""""Outputs the results for stage 1 and/or 2."""
import copy
from typing import Dict

from typeguard import typechecked

from snncompare.exp_config.Exp_config import Exp_config
from snncompare.run_config.Run_config import Run_config

from .helper import run_config_to_filename
from .Output import output_stage_json


@typechecked
def output_files_stage_1_and_2(
    *,
    exp_config: Exp_config,
    run_config: Run_config,
    results_nx_graphs: Dict,
    stage_index: int,
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

    # TODO: output should not modify results_nx_graphs. Currently it does
    # because it stores ..aphs[graph_name] = Json_dict_simsnn(simulator)
    output_stage_json(
        exp_config=exp_config,
        run_config=run_config,
        results_nx_graphs=copy.deepcopy(results_nx_graphs),
        run_config_filename=json_filename,
        stage_index=stage_index,
    )
