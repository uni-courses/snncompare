""""Stores the run config Dict type."""
from __future__ import annotations

from typeguard import typechecked


# pylint: disable=R0902
# pylint: disable=R0903
class Optional_config:
    """Stores optional configuration settings."""

    # pylint: disable=R0913
    # pylint: disable=R0914
    @typechecked
    def __init__(
        self,
        recreate_stages: list[int],
        export_types: list[str],
        gif: bool,
        zoom: Zoom,
        json_output_levels: Json_output_levels,
    ):
        """Stores run configuration settings for the exp_configriment."""
        self.recreate_stages: list[int] = recreate_stages
        self.export_types: list[str] = export_types
        self.gif: bool = bool(gif)
        self.zoom: Zoom = zoom
        if self.zoom.create_zoomed_image and "png" not in self.export_types:
            raise SyntaxError(
                "Error, zoomed images can only be created on png images. No "
                "png images were requested to be created."
            )

        self.json_output_levels = json_output_levels


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


class Json_output_levels:
    """Specify what information of the runs will be outputted to json files."""

    # pylint: disable=R0913
    @typechecked
    def __init__(
        self,
        inititial_graphs: bool,
        propagated_graphs: bool,
        results: bool,
        boxplot: bool,
        nr_of_spikes: bool,
        nr_of_neurons: bool,
        nr_of_synapses: bool,
        died_neurons: bool,
    ):
        self.inititial_graphs: bool = inititial_graphs
        if not self.inititial_graphs:
            raise Exception(
                "Error, not outputting inititial_graphs yet implemented."
            )
        self.propagated_graphs: bool = propagated_graphs
        if not self.propagated_graphs:
            raise Exception(
                "Error, not outputting propagated_graphs yet implemented."
            )
        self.results: bool = results
        if not self.results:
            raise Exception(
                "Error, not outputting results not yet implemented."
            )
        self.boxplot: bool = boxplot
        if self.boxplot:
            raise Exception("Error, boxplot not yet implemented.")
        self.nr_of_spikes: bool = nr_of_spikes
        if self.nr_of_spikes:
            raise Exception("Error, nr_of_spikes not yet implemented.")
        self.nr_of_neurons: bool = nr_of_neurons
        if self.nr_of_neurons:
            raise Exception("Error, nr_of_neurons not yet implemented.")
        self.nr_of_synapses: bool = nr_of_synapses
        if self.nr_of_synapses:
            raise Exception("Error, nr_of_synapses not yet implemented.")
        self.died_neurons: bool = died_neurons
        if self.nr_of_spikes:
            raise Exception("Error, nr_of_spikes not yet implemented.")
