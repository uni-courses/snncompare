"""Checks whether stage 1 has been outputted."""
from pathlib import Path
from typing import Optional

import networkx as nx
from typeguard import typechecked

from snncompare.export_results.output_stage1_configs_and_input_graph import (
    get_rand_nrs_and_hash,
)
from snncompare.graph_generation.stage_1_create_graphs import (
    get_input_graph_of_run_config,
)
from snncompare.import_results.helper import (
    get_isomorphic_graph_hash,
    get_radiation_description,
    simsnn_files_exists_and_get_path,
)
from snncompare.run_config.Run_config import Run_config


@typechecked
def has_outputted_stage_1(
    *,
    input_graph: nx.Graph,
    run_config: Run_config,
) -> bool:
    """Returns True if the:

    - radiation names
    - random numbers
    - snn graph
    - (optional) adapted snn graphs
    have been outputted for the isomorphic hash belonging to this run_config.
    """
    for with_adaptation in [False, True]:
        if not has_outputted_snn_graph(
            input_graph=input_graph,
            run_config=run_config,
            with_adaptation=with_adaptation,
            stage_index=1,
        ):
            return False
    return True


def has_outputted_input_graph(
    *, input_graph: nx.Graph, run_config: Run_config
) -> bool:
    """Returns True if the rand_nrs for this run config has been outputted."""
    isomorphic_hash: str = get_isomorphic_graph_hash(some_graph=input_graph)
    output_dir: str = f"results/input_graphs/{run_config.graph_size}/"
    output_filepath: str = f"{output_dir}{isomorphic_hash}.json"
    return Path(output_filepath).is_file()


def has_outputted_snn_graph(
    *,
    input_graph: nx.Graph,
    run_config: Run_config,
    with_adaptation: bool,
    stage_index: int,
    rad_affected_neurons_hash: Optional[str] = None,
) -> bool:
    """Returns True if the rand_nrs for this run config has been outputted."""
    _, rand_nrs_hash = get_rand_nrs_and_hash(input_graph=input_graph)
    simsnn_exists, _ = simsnn_files_exists_and_get_path(
        output_category="snns",
        input_graph=input_graph,
        run_config=run_config,
        with_adaptation=with_adaptation,
        stage_index=stage_index,
        rand_nrs_hash=rand_nrs_hash,
        rad_affected_neurons_hash=rad_affected_neurons_hash,
    )
    return simsnn_exists


def has_outputted_rand_nrs(
    *, input_graph: nx.Graph, run_config: Run_config
) -> bool:
    """Returns True if the rand_nrs for this run config has been outputted."""
    rand_nrs_exists, rand_nrs_filepath = simsnn_files_exists_and_get_path(
        output_category="rand_nrs",
        input_graph=input_graph,
        run_config=run_config,
        with_adaptation=False,
        stage_index=1,
    )
    if not rand_nrs_exists:
        return False
    # TODO: verify if file contains radiation neurons for this seed.
    raise FileNotFoundError(f"{rand_nrs_filepath} does not exist.")


def has_outputted_radiation(
    *, input_graph: nx.Graph, run_config: Run_config
) -> bool:
    """Returns True if the radiation for this run config has been outputted."""
    radiation_name, radiation_parameter = get_radiation_description(
        run_config=run_config
    )
    if radiation_name == "neuron_death":
        for with_adaptation in [False, True]:
            (
                radiation_file_exists,
                radiation_filepath,
            ) = simsnn_files_exists_and_get_path(
                output_category=f"{radiation_name}_{radiation_parameter}",
                input_graph=input_graph,
                run_config=run_config,
                with_adaptation=with_adaptation,
                stage_index=1,
            )
            if not radiation_file_exists:
                return False
            # TODO: verify if file contains radiation neurons for this seed.
            raise FileNotFoundError(f"{radiation_filepath} does not exist.")
    raise NotImplementedError(
        f"Error {radiation_name} is not yet implemented."
    )


@typechecked
def assert_has_outputted_stage_1(run_config: Run_config) -> None:
    """Throws error if stage 1 is not outputted."""
    if not has_outputted_stage_1(
        input_graph=get_input_graph_of_run_config(run_config=run_config),
        run_config=run_config,
    ):
        raise ValueError("Error, stage 1 was not completed.")
