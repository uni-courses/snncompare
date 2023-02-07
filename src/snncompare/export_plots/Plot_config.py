"""Generates interactive view of graph."""

from typeguard import typechecked


# pylint: disable=R0902
class Plot_config:
    """Stores the configuration for the plots."""

    # pylint: disable=R0903
    # pylint: disable=R0913
    @typechecked
    def __init__(
        self,
        base_pixel_width: int,
        base_pixel_height: int,
        node_size: float,
        edge_width_factor: float,
        recursive_edge_radius_factor: float,
        dx_redundant_factor: float,
        dy_redundant_factor: float,
        x_node_spacer_factor: float,
        y_node_spacer_factor: float,
        neuron_text_size_factor: float,
        edge_label_size_factor: float,
    ) -> None:
        """Creates the plot configuration to generate the svg files."""

        self.base_pixel_width: int = base_pixel_width
        self.base_pixel_height: int = base_pixel_height
        self.node_size: float = node_size
        self.edge_width: float = edge_width_factor * node_size
        self.recursive_edge_radius: float = (
            recursive_edge_radius_factor * node_size
        )
        self.dx_redundant: float = dx_redundant_factor * node_size
        self.dy_redundant: float = dy_redundant_factor * node_size
        self.x_node_spacer: float = x_node_spacer_factor * node_size
        self.y_node_spacer: float = y_node_spacer_factor * node_size
        self.neuron_text_size: float = neuron_text_size_factor * node_size
        self.edge_label_size: float = edge_label_size_factor * self.edge_width


def get_default_plot_config() -> Plot_config:
    """Returns a default Plot_config object."""
    plot_config: Plot_config = Plot_config(
        base_pixel_width=1000,
        base_pixel_height=1000,
        node_size=0.1,
        edge_width_factor=0.4,
        recursive_edge_radius_factor=0.1,
        dx_redundant_factor=0.2,
        dy_redundant_factor=0.2,
        x_node_spacer_factor=0.5,
        y_node_spacer_factor=0.5,
        neuron_text_size_factor=0.5,
        edge_label_size_factor=0.5,
    )
    return plot_config
