"""File used to create and export plots and tables directly into latex. Can be
used to automatically update your results each time you run latex.

For copy-pastable examples, see:     example_create_a_table()
example_create_multi_line_plot()     example_create_single_line_plot()
at the bottom of this file.
"""
import os
from typing import List

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import lines
from typeguard import typechecked


class Plot_to_tex:
    """Object used to output plots to latex directory of project."""

    @typechecked
    def __init__(self) -> None:
        self.script_dir = self.get_script_dir()

    # plot graph (legendPosition = integer 1 to 4)
    @typechecked
    def plotSingleLine(
        self,
        x_path: range,
        y_series: np.ndarray,
        x_axis_label: str,
        y_axis_label: str,
        label: str,
        filename: str,
        legendPosition: int,
    ) -> None:
        """# TODO: delete or update function.

        :param x_path: param y_series:
        :param x_axis_label: param y_axis_label:
        :param label: param filename:
        :param legendPosition:
        :param y_series: param y_axis_label:
        :
        """
        # pylint: disable=R0913
        # TODO: reduce 9/5 arguments to at most 5/5 arguments.
        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.plot(x_path, y_series, c="b", ls="-", label=label, fillstyle="none")
        plt.legend(loc=legendPosition)
        plt.xlabel(x_axis_label)
        plt.ylabel(y_axis_label)
        plt.savefig("latex/" + "/Images/" + filename + ".png")

        # plt.show()
        plt.clf()
        plt.close()

    # plot graphs
    @typechecked
    def plotMultipleLines(
        self,
        x: List,
        y_series: np.ndarray,
        x_label: str,
        y_label: str,
        label: List,
        filename: str,
        legendPosition: int,
    ) -> None:
        """# TODO: delete or update function.

        :param x: param y_series:
        :param x_label: param y_label:
        :param label: param filename:
        :param legendPosition:
        :param y_series: param y_label:
        :param filename:
        :param y_label:
        :
        """
        # pylint: disable=R0913
        # TODO: reduce 9/5 arguments to at most 5/5 arguments.
        fig = plt.figure()
        ax = fig.add_subplot(111)

        # generate colours
        print(f"typeself={self.__dict__}")
        cmap = get_cmap(plt, len(y_series[:, 0]))

        # generate line types
        lineTypes = generateLineTypes(y_series)

        for i in range(0, len(y_series)):
            # overwrite linetypes to single type
            lineTypes[i] = "-"
            ax.plot(
                x,
                y_series[i, :],
                ls=lineTypes[i],
                label=label[i],
                fillstyle="none",
                c=cmap(i),
            )
            # color

        # configure plot layout
        plt.legend(loc=legendPosition)
        plt.xlabel(x_label)
        plt.ylabel(y_label)
        plt.savefig(
            # os.path.dirname(__file__)
            "latex/"
            + "/Images/"
            + filename
            + ".png"
        )
        plt.clf()
        plt.close()

    # Create a table with: table_matrix = np.zeros((4,4),dtype=object) and pass
    # it to this object
    @typechecked
    def put_table_in_tex(
        self, table_matrix: np.ndarray, filename: str
    ) -> None:
        """This table can be directly plotted into latex by putting the
        commented code below into your latex file at the position where you
        want your table:"""
        # \begin{table}[H]
        #     \\centering
        #     \\caption{Results some computation.}\\label{tab:some_computation}
        #     \begin{tabular}{|c|c|} % remember to update this to show all
        #     %columns of table
        #         \\hline
        #         \\input{latex/project3/tables/q2.txt}
        #     \\end{tabular}
        # \\end{table}

        # You should update the number of columns in that latex code.

        cols = np.shape(table_matrix)[1]
        some_format = "%s"
        for _ in range(1, cols):
            some_format = some_format + " & %s"
        some_format = some_format + ""
        print(f"format={format}")
        # TODO: Change to something else to save as txt.
        os.mkdir("latex/tables/")
        np.savetxt(
            "latex/" + "tables/" + filename + ".txt",
            table_matrix,
            delimiter=" & ",
            # fmt=format,  # type: ignore[arg-type]
            fmt=some_format,  # type: ignore[arg-type]
            newline="  \\\\ \\hline \n",
        )

    @typechecked
    def export_plot(
        self, some_plt: matplotlib.pyplot, filename: str, extensions: List[str]
    ) -> None:
        """

        :param plt:
        :param filename:

        """
        self.create_target_dir_if_not_exists("latex/Images/", "graphs")
        for extension in extensions:
            some_plt.savefig(
                "latex/Images/" + "graphs/" + filename + f".{extension}",
                dpi=200,
            )

    @typechecked
    def get_script_dir(self) -> str:
        """returns the directory of this script regardless of from which level
        the code is executed."""
        return os.path.dirname(__file__)

    @typechecked
    def create_target_dir_if_not_exists(
        self, path: str, new_dir_name: str
    ) -> None:
        """

        :param path:
        :param new_dir_name:

        """
        if os.path.exists(path):
            if not os.path.exists(f"{path}/{new_dir_name}"):
                os.makedirs(f"{path}/{new_dir_name}")
        else:
            raise Exception(f"Error, path={path} did not exist.")


# Generate random line colours
# Source: https://stackoverflow.com/questions/14720331/
# how-to-generate-random-colors-in-matplotlib
@typechecked
def get_cmap(
    some_plt: matplotlib.pyplot,
    nr_of_colours: int,
    name: str = "hsv",
) -> matplotlib.colors.LinearSegmentedColormap:
    """Returns a function that maps each index in 0, 1, ..., n-1 to a distinct
    RGB color; the keyword argument name must be a standard mpl colormap name.

    :param n: param name:  (Default value = "hsv")
    :param name: Default value = "hsv")
    """
    return some_plt.cm.get_cmap(name, nr_of_colours)


@typechecked
def generateLineTypes(y_series: np.ndarray) -> List:
    """

    :param y_series:

    """
    # generate varying linetypes
    typeOfLines = list(lines.lineStyles.keys())

    while len(y_series) > len(typeOfLines):
        typeOfLines.append("-.")

    # remove void lines
    for i in range(0, len(y_series)):
        if typeOfLines[i] == "None":
            typeOfLines[i] = "-"
        if typeOfLines[i] == "":
            typeOfLines[i] = ":"
        if typeOfLines[i] == " ":
            typeOfLines[i] = "--"
    return typeOfLines


# replace this with your own table creation and then pass it to
# put_table_in_tex(..)
@typechecked
def example_create_a_table() -> None:
    """Example on how to create a latex table from Python."""
    # pylint: disable=C0415
    from snncompare.export_plots.Plot_to_tex import Plot_to_tex as plt_tex

    table_name = "example_table_name"
    rows = 2
    columns = 4
    table_matrix = np.zeros((rows, columns), dtype=object)
    table_matrix[:, :] = ""  # replace the standard zeros with empty cell
    for column in range(0, columns):
        for row in range(0, rows):
            table_matrix[row, column] = row + column
    table_matrix[1, 0] = "example"
    table_matrix[0, 1] = "grid sizes"

    plt_tex.put_table_in_tex(
        self=plt_tex, table_matrix=table_matrix, filename=table_name
    )


@typechecked
def example_create_multi_line_plot() -> None:
    """Example that creates a plot with multiple lines.

    Copy paste it in your own code and modify the values accordingly.
    """
    # pylint: disable=C0415
    from snncompare.export_plots.Plot_to_tex import Plot_to_tex as plt_tex

    multiple_y_series = np.zeros((2, 2), dtype=int)
    # actually fill with data
    multiple_y_series[0] = [1, 2]
    lineLabels = [
        "first-line",
        "second_line",
    ]  # add a label for each dataseries
    single_x_series = [3, 5]

    plt_tex.plotMultipleLines(
        self=plt_tex,
        x=single_x_series,
        y_series=multiple_y_series,
        x_label="x-axis label [units]",
        y_label="y-axis label [units]",
        label=lineLabels,
        filename="3b",
        legendPosition=4,
    )


@typechecked
def example_create_single_line_plot() -> None:
    """Example that creates a plot with a single line.

    Copy paste it in your own code and modify the values accordingly.
    """
    # pylint: disable=C0415
    from snncompare.export_plots.Plot_to_tex import Plot_to_tex as plt_tex

    multiple_y_series = np.zeros((2, 2), dtype=int)
    # actually fill with data
    multiple_y_series[0] = [1, 2]
    lineLabels = [
        "first-line",
        "second_line",
    ]  # add a label for each dataseries
    single_x_series = [3, 5]

    plt_tex.plotMultipleLines(
        self=plt_tex,
        x=single_x_series,
        y_series=multiple_y_series,
        x_label="x-axis label [units]",
        y_label="y-axis label [units]",
        label=lineLabels,
        filename="3b",
        legendPosition=4,
    )


if __name__ == "__main__":
    example_create_a_table()
    example_create_multi_line_plot()
    example_create_single_line_plot()
