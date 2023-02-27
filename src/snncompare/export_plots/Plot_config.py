"""Generates interactive view of graph."""


from typeguard import typechecked


# pylint: disable=R0902
class Plot_config:
    """Stores the configuration for the plots."""

    # pylint: disable=R0903
    # pylint: disable=R0913
    # pylint: disable=R0914
    @typechecked
    def __init__(
        self,
        base_pixel_width: int,
        base_pixel_height: int,
        dx_redundant_factor: float,
        dy_redundant_factor: float,
        edge_width_factor: float,
        edge_label_size_factor: float,
        neuron_text_size_factor: float,
        node_size: float,
        recursive_edge_radius_factor: float,
        show_nodes: bool,
        show_node_colours: bool,
        show_node_labels: bool,
        show_node_opacity: bool,
        show_edge_colours: bool,
        show_edge_labels: bool,
        show_edge_opacity: bool,
        show_x_ticks: bool,
        show_y_ticks: bool,
        update_node_colours: bool,
        update_node_labels: bool,
        update_node_opacity: bool,
        update_edge_colours: bool,
        update_edge_labels: bool,
        update_edge_opacity: bool,
        x_node_spacer_factor: float,
        x_tick_size_factor: float,
        y_node_spacer_factor: float,
        y_tick_size_factor: float,
    ) -> None:
        """Creates the plot configuration to generate the svg files."""

        self.base_pixel_width: int = base_pixel_width
        self.base_pixel_height: int = base_pixel_height
        self.node_size: float = node_size
        self.edge_width: float = edge_width_factor * node_size
        if self.edge_width < 1:
            raise ValueError(
                "Error, edge width should be larger than 1. it "
                + f"is:{self.edge_width}"
            )

        self.recursive_edge_radius: float = (
            recursive_edge_radius_factor * node_size
        )
        self.dx_redundant: float = dx_redundant_factor * node_size
        self.dy_redundant: float = dy_redundant_factor * node_size
        self.x_node_spacer: float = x_node_spacer_factor * node_size
        self.y_node_spacer: float = y_node_spacer_factor * node_size
        self.neuron_text_size: float = neuron_text_size_factor * node_size
        if self.neuron_text_size < 1:
            raise ValueError(
                "Error, edge width should be larger than 1. it "
                + f"is:{self.neuron_text_size}"
            )
        self.edge_label_size: float = edge_label_size_factor * self.edge_width
        self.x_tick_size: float = x_tick_size_factor
        self.y_tick_size: float = y_tick_size_factor

        self.show_nodes: bool = show_nodes
        self.show_node_colours: bool = show_node_colours
        self.show_node_labels: bool = show_node_labels
        self.show_node_opacity: bool = show_node_opacity
        self.show_edge_colours: bool = show_edge_colours
        self.show_edge_labels: bool = show_edge_labels
        self.show_edge_opacity: bool = show_edge_opacity
        self.show_x_ticks: bool = show_x_ticks
        self.show_y_ticks: bool = show_y_ticks
        self.update_node_colours: bool = update_node_colours
        self.update_node_labels: bool = update_node_labels
        self.update_node_opacity: bool = update_node_opacity
        self.update_edge_colours: bool = update_edge_colours
        self.update_edge_labels: bool = update_edge_labels
        self.update_edge_opacity: bool = update_edge_opacity


def get_default_plot_config() -> Plot_config:
    """Returns a default Plot_config object."""
    plot_config: Plot_config = Plot_config(
        base_pixel_width=100,
        base_pixel_height=100,
        node_size=5,
        edge_width_factor=0.2,
        recursive_edge_radius_factor=0.02,
        show_nodes=True,
        show_node_colours=False,
        show_node_labels=False,
        show_node_opacity=False,
        show_edge_colours=True,
        show_edge_labels=False,
        show_edge_opacity=True,
        show_x_ticks=False,
        show_y_ticks=False,
        update_node_colours=False,
        update_node_labels=True,
        update_node_opacity=False,
        update_edge_colours=True,
        update_edge_labels=False,
        update_edge_opacity=True,
        dx_redundant_factor=0.05,
        dy_redundant_factor=0.05,
        x_node_spacer_factor=0.15,
        y_node_spacer_factor=0.15,
        neuron_text_size_factor=0.8,
        edge_label_size_factor=0.8,
        x_tick_size_factor=15,
        y_tick_size_factor=15,
    )
    return plot_config
