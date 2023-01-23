"""Creates the experiment results in the form of plots with lines indicating
the performance of the SNNs."""

# Take in exp_config or run_configs
# If exp_config, get run_configs
from typing import Dict, List, Tuple

from typeguard import typechecked

from snncompare.exp_config.Exp_config import Exp_config
from snncompare.exp_config.run_config.Run_config import Run_config
from snncompare.Experiment_runner import Experiment_runner
from snncompare.export_results.load_json_to_nx_graph import (
    load_verified_json_graphs_from_json,
)
from snncompare.helper import generate_run_configs
from snncompare.import_results.check_completed_stages import (
    has_outputted_stage,
)


@typechecked
def create_performance_plots(exp_config: Exp_config) -> None:
    """Ensures all performance boxplots are created."""

    # Determine which lines are plotted, e.g.
    # which algorithm with/without adaptation.

    # Get run_configs
    run_configs = generate_run_configs(exp_config, specific_run_config=None)

    # Get list of required run_configs.
    (
        completed_run_configs,
        missing_run_configs,
    ) = get_completed_and_missing_run_configs(run_configs)

    # Verify all json results are available.

    # If not, create a list of run_configs that still need to be completed.
    # prompt user whether user wants to complete these run_configs.
    for missing_run_config in missing_run_configs:
        # Execute those run_configs
        exp_runner = Experiment_runner(
            exp_config=exp_config,
            specific_run_config=missing_run_config,
            perform_run=True,
        )
        # Terminate code.

    print(completed_run_configs)
    print(exp_runner)
    # Get results per line.

    # Generate line plots
    # Generate box plots.

    # Allow user to select subset of parameter ranges.
    # (This is done in the experiment setting)
    # graph_sizes 10-30
    # m_values 0 to 10
    # adaptation 0-4
    # radiation 0,0.1,0.25

    # Get results per data

    # Create dummy box plots.

    # Create dummy line plots.


@typechecked
def get_json_data(run_configs: List[Run_config]) -> Dict:
    """Loads the data from the relevant .json dicts and returns it."""
    run_config_nx_graphs: Dict[Run_config, Dict] = {}
    for run_config in run_configs:
        run_config_nx_graphs[run_config] = load_verified_json_graphs_from_json(
            run_config=run_config,
            expected_stages=[1, 2, 4],
        )

    # TODO: write logif if file does not exist.
    return run_config_nx_graphs


@typechecked
def get_completed_and_missing_run_configs(
    run_configs: List[Run_config],
) -> Tuple[List[Run_config], List[Run_config]]:
    """Returns the run configs that still need to be ran."""
    missing_run_configs: List[Run_config] = []
    completed_run_configs: List[Run_config] = []
    for run_config in run_configs:
        if not has_outputted_stage(
            run_config=run_config,
            stage_index=4,
            to_run={
                "stage_1": True,
                "stage_2": True,
                "stage_3": False,
                "stage_4": True,
            },
        ):
            missing_run_configs.append(run_config)
        else:
            completed_run_configs.append(run_config)
    print(f"Want:{len(run_configs)}, missing:{len(missing_run_configs)}")
    return completed_run_configs, missing_run_configs


@typechecked
def get_boxplot_datapoints(
    run_config_nx_graphs: Dict,
) -> List[List[int]]:
    """Returns the run configs that still need to be ran."""

    # Create x-axis categories (no redundancy, n-redundancy).
    for run_config, graphs_dict in run_config_nx_graphs.items():
        # Get the results per x-axis category per graph type.
        print(f"TODO: {run_config},{graphs_dict}")

    # return data.
    return [[23, 40], [42, 44]]
