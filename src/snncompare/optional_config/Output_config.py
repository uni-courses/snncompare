""""Stores the run config Dict type."""
from __future__ import annotations

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
    ):
        """Stores run configuration settings for the exp_configriment."""
        self.verify_int_list_values(
            input_list=output_json_stages,
            permitted_list=[1, 2, 4, 5],
            param_name="recreate_stages",
        )

        self.verify_output_json_stages(
            output_json_stages=output_json_stages,
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

        self.zoom: Zoom = zoom
        if self.zoom.create_zoomed_image and "png" not in self.export_types:
            raise SyntaxError(
                "Error, zoomed images can only be created on png images. No "
                "png images were requested to be created."
            )

        self.extra_storing_config: Extra_storing_config = extra_storing_config

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

    @typechecked
    def verify_output_json_stages(
        self,
        output_json_stages: list[int],
    ) -> None:
        """TODO: remove this checks and implement the functionalities."""
        if 1 not in output_json_stages:
            raise Exception(
                "Error, not outputting inititial_graphs yet implemented."
            )
        if 2 not in output_json_stages:
            raise Exception(
                "Error, not outputting propagated_graphs yet implemented."
            )

        if 4 not in output_json_stages:
            raise Exception(
                "Error, not outputting results not yet implemented."
            )

        if 5 in output_json_stages:
            raise Exception("Error, boxplot not yet implemented.")


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
        store_died_neurons: bool,  # Does this say dead neuron storage?
    ):
        self.count_spikes: bool = count_spikes
        if self.count_spikes:
            raise Exception("Error, count_spikes not yet implemented.")
        self.count_neurons: bool = count_neurons
        if self.count_neurons:
            raise Exception("Error, count_neurons not yet implemented.")
        self.count_synapses: bool = count_synapses
        if self.count_synapses:
            raise Exception("Error, count_synapses not yet implemented.")
        self.store_died_neurons: bool = store_died_neurons
        if self.count_spikes:
            raise Exception("Error, count_spikes not yet implemented.")
