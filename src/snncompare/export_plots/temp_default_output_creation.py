"""Temporary duplicate code from tests because they were not included in pip
package.

TODO: move the duplicate code out of tests and rename this file.
"""
from typeguard import typechecked

from snncompare.exp_config.Exp_config import Exp_config
from snncompare.optional_config.Output_config import (
    Extra_storing_config,
    Hover_info,
    Output_config,
    Zoom,
)


@typechecked
def create_default_output_config(*, exp_config: Exp_config) -> Output_config:
    """Create duplicate Output_config that is used to generate the data
    belonging to each run config, using the Experiment runner."""
    output_config = Output_config(
        recreate_stages=[1, 2, 4],
        export_types=[],
        zoom=Zoom(
            create_zoomed_image=False,
            left_right=None,
            bottom_top=None,
        ),
        output_json_stages=[1, 2, 4],
        hover_info=create_default_hover_info(exp_config=exp_config),
        extra_storing_config=Extra_storing_config(
            count_spikes=False,
            count_neurons=False,
            count_synapses=False,
            skip_stage_2_output=True,
            show_images=True,
            store_died_neurons=False,
        ),
    )
    return output_config


@typechecked
def create_default_hover_info(*, exp_config: Exp_config) -> Hover_info:
    """Create duplicate Hover_info that is used to generate the data belonging
    to each run config, using the Experiment runner."""

    hover_info = Hover_info(
        incoming_synapses=True,
        neuron_models=exp_config.neuron_models,
        neuron_properties=[
            "spikes",
            "a_in",
            "bias",
            "du",
            "u",
            "dv",
            "v",
            "vth",
        ],
        node_names=True,
        outgoing_synapses=True,
        synaptic_models=exp_config.synaptic_models,
        synapse_properties=["weight"],
    )
    return hover_info
