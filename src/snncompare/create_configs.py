"""Contains helper functions that are used throughout this repository."""
from pprint import pprint
from typing import Dict, List, Optional, Tuple, Union

# from snncompare.export_results.load_json_to_nx_graph import dicts_are_equal
from typeguard import typechecked

from snncompare.exp_config.Exp_config import Exp_config
from snncompare.helper import dicts_are_equal
from snncompare.run_config.Run_config import Run_config

# if TYPE_CHECKING:
# from snncompare.exp_config.Exp_config import Exp_config


@typechecked
def generate_run_configs(
    *,
    exp_config: "Exp_config",
    specific_run_config: Optional[Run_config] = None,
) -> List[Run_config]:
    """Generates the run configs belonging to an experiment config, and then
    removes all run configs except for the desired run config.

    Throws an error if the desired run config is not within the expected
    run configs.
    """
    found_run_config = False
    pprint(exp_config.__dict__)
    # Generate run configurations.
    run_configs: List[Run_config] = exp_config_to_run_configs(
        exp_config=exp_config
    )
    if specific_run_config is not None:
        for gen_run_config in run_configs:
            if dicts_are_equal(
                left=gen_run_config.__dict__,
                right=specific_run_config.__dict__,
                without_unique_id=True,
            ):
                found_run_config = True
                if gen_run_config.unique_id != specific_run_config.unique_id:
                    raise ValueError(
                        "Error, equal dict but unequal unique_ids."
                    )
                break

        if not found_run_config:
            pprint(run_configs)
            raise ValueError(
                f"The expected run config:{specific_run_config} was not"
                "found."
            )
        run_configs = [specific_run_config]

    return run_configs


@typechecked
def exp_config_to_run_configs(
    *,
    exp_config: "Exp_config",
) -> List[Run_config]:
    """Generates all the run_config dictionaries of a single experiment
    configuration. Then verifies whether each run_config is valid.

    TODO: Ensure this can be done modular, and lower the depth of the loop.
    """
    # pylint: disable=R0914

    run_configs: List[Run_config] = []

    # pylint: disable=R1702
    # TODO: make it loop through a list of keys.
    # for algorithm in exp_config.algorithms:
    for algorithm_name, algo_specs in exp_config.algorithms.items():
        for algo_config in algo_specs:
            algorithm = {algorithm_name: algo_config}

            for adaptation, radiation in get_adaptation_and_radiations(
                exp_config=exp_config
            ):
                fill_remaining_run_config_settings(
                    adaptation=adaptation,
                    algorithm=algorithm,
                    exp_config=exp_config,
                    radiation=radiation,
                    run_configs=run_configs,
                )
    return list(reversed(run_configs))


@typechecked
def fill_remaining_run_config_settings(
    *,
    adaptation: Union[None, Dict],
    algorithm: Dict,
    exp_config: "Exp_config",
    radiation: Union[None, Dict],
    run_configs: List[Run_config],
) -> None:
    """Generate basic settings for a run config."""
    for seed in exp_config.seeds:
        for size_and_max_graph in exp_config.size_and_max_graphs:
            for simulator in exp_config.simulators:
                for graph_nr in range(0, size_and_max_graph[1]):
                    run_config: Run_config = run_parameters_to_dict(
                        adaptation=adaptation,
                        algorithm=algorithm,
                        seed=seed,
                        size_and_max_graph=size_and_max_graph,
                        graph_nr=graph_nr,
                        radiation=radiation,
                        simulator=simulator,
                    )
                    run_configs.append(run_config)


# pylint: disable=R0913
@typechecked
def run_parameters_to_dict(
    *,
    adaptation: Union[None, Dict[str, int]],
    algorithm: Dict[str, Dict[str, int]],
    seed: int,
    size_and_max_graph: Tuple[int, int],
    graph_nr: int,
    radiation: Union[None, Dict],
    simulator: str,
) -> Run_config:
    """Stores selected parameters into a dictionary.

    Written as separate argument to keep code width under 80 lines. #
    TODO: verify typing.
    """
    run_config: Run_config = Run_config(
        adaptation=adaptation,
        algorithm=algorithm,
        seed=seed,
        graph_size=size_and_max_graph[0],
        graph_nr=graph_nr,
        radiation=radiation,
        simulator=simulator,
    )

    return run_config


def get_adaptation_and_radiations(
    *,
    exp_config: "Exp_config",
) -> List[tuple]:
    """Returns a list of adaptations and radiations that will be used for the
    experiment."""

    adaptations_radiations: List[tuple] = []
    if exp_config.adaptations is None:
        adaptation = None
        adaptations_radiations.extend(
            get_radiations(exp_config=exp_config, adaptation=adaptation)
        )
    else:
        for (
            adaptation_name,
            adaptation_setts_list,
        ) in exp_config.adaptations.items():
            for adaptation_config in adaptation_setts_list:
                adaptation = {adaptation_name: adaptation_config}
                adaptations_radiations.extend(
                    get_radiations(
                        exp_config=exp_config, adaptation=adaptation
                    )
                )

    # Make sure it contains at least 1 empty entry:
    if not adaptations_radiations:  # Empty list evaluates to False.
        return [(None, None)]
    return adaptations_radiations


def get_radiations(
    *, exp_config: "Exp_config", adaptation: Union[None, Dict[str, int]]
) -> List[Tuple[Union[None, Dict], Union[None, Dict]]]:
    """Returns the radiations."""
    adaptation_and_radiations: List[
        Tuple[Union[None, Dict], Union[None, Dict]]
    ] = []
    if exp_config.radiations is None:
        adaptation_and_radiations.append((adaptation, None))
    else:
        for (
            radiation_name,
            radiation_setts_list,
        ) in exp_config.radiations.items():
            # TODO: verify it is of type list.
            for rad_config in radiation_setts_list:
                radiation = {radiation_name: rad_config}
                adaptation_and_radiations.append((adaptation, radiation))
    return adaptation_and_radiations
