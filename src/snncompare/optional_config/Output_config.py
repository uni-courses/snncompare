""""Stores the run config Dict type."""
from __future__ import annotations

from snnbackends.networkx.LIF_neuron import LIF_neuron, Synapse
from typeguard import typechecked

from snncompare.exp_config.Exp_config import Supported_experiment_settings


# pylint: disable=R0902
# pylint: disable=R0903
class Output_config:
    """Stores optional configuration settings."""

    # pylint: disable=R0913
    # pylint: disable=R0914
    @typechecked
    def __init__(
        self,
        recreate_stages: list[int],
        export_types: list[str],
        zoom: Zoom,
        output_json_stages: list[int],
        extra_storing_config: Extra_storing_config,
        hover_info: Hover_info | None = None,
        graph_types: list[str] | None = None,
        dash_port: int | None = None,
    ):
        """Stores run configuration settings for the exp_configriment."""
        self.verify_int_list_values(
            input_list=output_json_stages,
            permitted_list=[1, 2, 4, 5, 6],
            param_name="recreate_stages",
        )

        self.output_json_stages = output_json_stages

        self.verify_int_list_values(
            input_list=recreate_stages,
            permitted_list=[1, 2, 3, 4, 5],
            param_name="recreate_stages",
        )
        self.recreate_stages: list[int] = recreate_stages

        self.verify_export_types(export_types)
        self.export_types: list[str] = export_types

        if hover_info is not None:
            self.hover_info: Hover_info = hover_info

        self.zoom: Zoom = zoom
        if self.zoom.create_zoomed_image and "png" not in self.export_types:
            raise SyntaxError(
                "Error, zoomed images can only be created on png images. No "
                "png images were requested to be created."
            )

        self.extra_storing_config: Extra_storing_config = extra_storing_config
        self.graph_types: None | list[str] = graph_types
        self.dash_port: None | int = dash_port

    @typechecked
    def verify_int_list_values(
        self,
        input_list: list[int],
        permitted_list: list[int],
        param_name: str,
    ) -> None:
        """Verifies the values in the recreate_stages list is valid."""
        for elem in input_list:
            if elem not in permitted_list:
                raise ValueError(
                    f"Error, {param_name}:{elem} not in supported"
                    + f" stages:{permitted_list}."
                )

    @typechecked
    def verify_export_types(
        self,
        export_types: list[str],
    ) -> None:
        """Verifies the export types are supported."""
        supp_setts = Supported_experiment_settings()
        for export_type in export_types:
            if export_type not in supp_setts.export_types:
                raise ValueError(
                    f"Error, export_type:{export_type} not in supported"
                    + f" export types:{supp_setts.export_types}."
                )


class Zoom:
    """Stores whether zoomed in images of png files will be created or not."""

    @typechecked
    def __init__(
        self,
        create_zoomed_image: bool,
        left_right: tuple[float, float] | None,
        bottom_top: tuple[float, float] | None,
    ):
        """The fractions of the image at which the zoomed region starts and
        ends."""
        if create_zoomed_image:
            if left_right is None:
                raise ValueError("Error, left_right not specified.")
            if bottom_top is None:
                raise ValueError("Error, bottom_top not specified.")
            for frac in [i for sub in [left_right, bottom_top] for i in sub]:
                print(f"frac={frac}")
                if frac < 0 or frac > 1:
                    raise ValueError(
                        "Error, zoom coordinates must be in range [0,1]."
                    )
            if left_right[0] >= left_right[1]:
                raise ValueError(
                    "Error, left fraction must me smaller than right fraction."
                )
            if bottom_top[0] >= bottom_top[1]:
                raise ValueError(
                    "Error, bottom fraction must me smaller than top fraction."
                )
            self.zoom_coords: Zoom_coords = Zoom_coords(
                x_left=left_right[0],
                x_right=left_right[1],
                y_bottom=bottom_top[0],
                y_top=bottom_top[1],
            )
            self.create_zoomed_image = True
        else:
            self.create_zoomed_image = False


class Zoom_coords:
    """The fractions of the image at which the zoomed region starts and
    ends."""

    # pylint: disable=R0913
    # pylint: disable=R0914
    @typechecked
    def __init__(
        self,
        x_left: float,
        x_right: float,
        y_bottom: float,
        y_top: float,
    ):
        self.x_left: float = x_left
        self.x_right: float = x_right
        self.y_bottom: float = y_bottom
        self.y_top: float = y_top


class Extra_storing_config:
    """Specify what information of the runs will be outputted to json files."""

    # pylint: disable=R0913
    @typechecked
    def __init__(
        self,
        count_spikes: bool,
        count_neurons: bool,
        count_synapses: bool,
        skip_stage_2_output: bool,
        show_images: bool,
        store_died_neurons: bool,
        export_failure_modes: bool,
        show_failure_modes: bool,
    ):
        self.count_spikes: bool = count_spikes
        self.skip_stage_2_output: bool = skip_stage_2_output
        if self.count_spikes:
            raise NotImplementedError(
                "Error, count_spikes not yet implemented."
            )
        self.count_neurons: bool = count_neurons
        if self.count_neurons:
            raise NotImplementedError(
                "Error, count_neurons not yet implemented."
            )
        self.count_synapses: bool = count_synapses
        if self.count_synapses:
            raise NotImplementedError(
                "Error, count_synapses not yet implemented."
            )
        self.export_failure_modes: bool = export_failure_modes
        self.show_images: bool = show_images
        self.store_died_neurons: bool = store_died_neurons
        self.show_failure_modes: bool = show_failure_modes
        if self.count_spikes:
            raise NotImplementedError(
                "Error, count_spikes not yet implemented."
            )


class Hover_info:
    """Specify what information is shown in dash when your mouse hovers over a
    node or synapse."""

    # pylint: disable=R0913
    @typechecked
    def __init__(
        self,
        incoming_synapses: bool,
        neuron_models: list[str],
        neuron_properties: list[str],
        node_names: bool,
        outgoing_synapses: bool,
        synaptic_models: list[str],
        synapse_properties: list[str],
    ):
        self.incoming_synapses: bool = incoming_synapses
        self.node_names: bool = node_names
        self.outgoing_synapses: bool = outgoing_synapses

        self.neuron_properties: list[str] = neuron_properties
        self.synapse_properties: list[str] = synapse_properties

        self.verify_requested_neuron_properties_exist(
            neuron_models=neuron_models,
            neuron_properties=neuron_properties,
        )
        self.verify_requested_synapse_properties_exist(
            synaptic_models=synaptic_models,
            synapse_properties=synapse_properties,
        )

    @typechecked
    def verify_requested_neuron_properties_exist(
        self,
        neuron_models: list[str],
        neuron_properties: list[str],
    ) -> None:
        """Verifies for each neuron model that the requested properties can be
        printed."""
        for neuron_model in neuron_models:
            if neuron_model == "LIF":
                sample_neuron: LIF_neuron = LIF_neuron(
                    name="dummy",
                    bias=0.1,
                    du=0.1,
                    dv=0.1,
                    vth=0.1,
                )
            else:
                raise NotImplementedError(
                    f"Error, showing neuron properties of type:{neuron_model} "
                    + "is not yet supported."
                )
            self.verify_neuron_has_properties_as_attributes(
                neuron_properties=neuron_properties,
                sample_neuron=sample_neuron,
            )

    @typechecked
    def verify_requested_synapse_properties_exist(
        self,
        synaptic_models: list[str],
        synapse_properties: list[str],
    ) -> None:
        """Verifies for each synaptic model that the requested properties can
        be printed."""
        for synaptic_model in synaptic_models:
            if synaptic_model == "LIF":
                sample_synapse: Synapse = Synapse(
                    weight=1,
                    delay=1,
                    change_per_t=1,
                )
            else:
                raise NotImplementedError(
                    "Error, showing synapse properties of type: "
                    + f"{synaptic_model} is not yet supported."
                )

            self.verify_synaptse_has_properties_as_attributes(
                synapse_properties=synapse_properties,
                sample_synapse=sample_synapse,
            )

    @typechecked
    def verify_neuron_has_properties_as_attributes(
        self,
        neuron_properties: list[str],
        sample_neuron: LIF_neuron,
    ) -> None:
        """Throws an error if the user asks to show neuron properties in dash
        that are not in the neuron attributes."""
        for neuron_property in neuron_properties:
            if neuron_property not in sample_neuron.__dict__.keys():
                raise KeyError(
                    f"Error, {neuron_property} does not exist in neuron "
                    f"attributes:{sample_neuron.__dict__.keys()}"
                )

    @typechecked
    def verify_synaptse_has_properties_as_attributes(
        self,
        synapse_properties: list[str],
        sample_synapse: Synapse,
    ) -> None:
        """Throws an error if the user asks to show synapse properties in dash
        that are not in the synapse attributes."""
        for synapse_property in synapse_properties:
            if synapse_property not in sample_synapse.__dict__.keys():
                raise KeyError(
                    f"Error, {synapse_property} does not exist in synapse "
                    f"attributes:{sample_synapse.__dict__.keys()}"
                )
