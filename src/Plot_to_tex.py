"""File used to export plots to a latex directory.
Call this from another file, for project 11, question 3b:
from Plot_to_tex import Plot_to_tex as plt_tex
multiple_y_series = np.zeros((nrOfDataSeries,nrOfDataPoints), dtype=int);
# actually fill with data
lineLabels = [] # add a label for each dataseries
plt_tex.plotMultipleLines(plt_tex,single_x_series,multiple_y_series,"x-axis
 label [units]","y-axis label [units]",lineLabels,"3b",4,11)
4b=filename
4 = position of legend, e.g. top right.
For a single line, use:
plt_tex.plotSingleLine(plt_tex,range(0, len(dataseries)),dataseries,"x-axis
label [units]","y-axis label [units]",lineLabel,"3b",4,11)"""

import os

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import lines

# You can also plot a table directly into latex, see example_create_a_table(..)
# Then put it in latex with for example:
# \begin{table}[H]
#    \centering
#    \caption{Results some computation.}\label{tab:some_computation}
#    \begin{tabular}{|c|c|} % remember to update this
# % to show all columns of table
#        \hline
#        \input{latex/project3/tables/q2.txt}
#    \end{tabular}
# \end{table}


class Plot_to_tex:
    """Object used to output plots to latex directory of project."""

    def __init__(self):
        self.script_dir = self.get_script_dir()

    # plot graph (legendPosition = integer 1 to 4)
    def plotSingleLine(
        self,
        x_path,
        y_series,
        x_axis_label,
        y_axis_label,
        label,
        filename,
        legendPosition,
        project_name,
    ):
        """

        :param x_path: param y_series:
        :param x_axis_label: param y_axis_label:
        :param label: param filename:
        :param legendPosition: param project_name:
        :param y_series: param y_axis_label:
        :param filename: param project_name:
        :param y_axis_label: param project_name:
        :param project_name:

        """
        # pylint: disable=R0913
        # TODO: reduce 9/5 arguments to at most 5/5 arguments.
        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.plot(x_path, y_series, c="b", ls="-", label=label, fillstyle="none")
        plt.legend(loc=legendPosition)
        plt.xlabel(x_axis_label)
        plt.ylabel(y_axis_label)
        plt.savefig(
            os.path.dirname(__file__)
            + f"/../../../latex/{project_name}"
            + "/Images/"
            + filename
            + ".png"
        )

    #         plt.show();

    # plot graphs
    def plotMultipleLines(
        self,
        x,
        y_series,
        x_label,
        y_label,
        label,
        filename,
        legendPosition,
        project_name,
    ):
        """

        :param x: param y_series:
        :param x_label: param y_label:
        :param label: param filename:
        :param legendPosition: param project_name:
        :param y_series: param y_label:
        :param filename: param project_name:
        :param y_label: param project_name:
        :param project_name:

        """
        # pylint: disable=R0913
        # TODO: reduce 9/5 arguments to at most 5/5 arguments.
        fig = plt.figure()
        ax = fig.add_subplot(111)

        # generate colours
        cmap = self.get_cmap(len(y_series[:, 0]))

        # generate line types
        lineTypes = self.generateLineTypes(y_series)

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
            os.path.dirname(__file__)
            + f"/../../../latex/{project_name}"
            + "/Images/"
            + filename
            + ".png"
        )

    # Generate random line colours
    # Source: https://stackoverflow.com/questions/14720331/
    # how-to-generate-random-colors-in-matplotlib
    def get_cmap(self, n, name="hsv"):
        """Returns a function that maps each index in 0, 1, ..., n-1 to a
        distinct RGB color; the keyword argument name must be a standard mpl
        colormap name.

        :param n: param name:  (Default value = "hsv")
        :param name: Default value = "hsv")
        """
        return plt.cm.get_cmap(name, n)

    def generateLineTypes(self, y_series):
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

    # Create a table with: table_matrix = np.zeros((4,4),dtype=object) and pass
    # it to this object
    def put_table_in_tex(self, table_matrix, filename, project_name):
        """

        :param table_matrix: param filename:
        :param project_name: param filename:
        :param filename:

        """
        cols = np.shape(table_matrix)[1]
        some_format = "%s"
        for _ in range(1, cols):
            some_format = some_format + " & %s"
        some_format = some_format + ""
        # TODO: Change to something else to save as txt.
        np.savetxt(
            os.path.dirname(__file__)
            + f"/../../../latex/{project_name}"
            + "/tables/"
            + filename
            + ".txt",
            table_matrix,
            delimiter=" & ",
            fmt=format,
            newline="  \\\\ \\hline \n",
        )

    # replace this with your own table creation and then pass it to
    # put_table_in_tex(..)
    def example_create_a_table(self):
        """Example on how to create a latex table from Python."""
        project_name = "1"
        table_name = "example_table_name"
        rows = 2
        columns = 4
        table_matrix = np.zeros((rows, columns), dtype=object)
        table_matrix[:, :] = ""  # replace the standard zeros with empty cell
        print(table_matrix)
        for column in range(0, columns):
            for row in range(0, rows):
                table_matrix[row, column] = row + column
        table_matrix[1, 0] = "example"
        table_matrix[0, 1] = "grid sizes"

        self.put_table_in_tex(table_matrix, table_name, project_name)

    def get_script_dir(self):
        """returns the directory of this script regardless of from which level
        the code is executed."""
        return os.path.dirname(__file__)

    def export_plot(self, some_plt, filename):
        """

        :param plt:
        :param filename:

        """
        self.create_target_dir_if_not_exists("latex/Images/", "graphs")
        some_plt.savefig(
            "latex/Images/" + "graphs/" + filename + ".png", dpi=200
        )

    def create_target_dir_if_not_exists(self, path, new_dir_name):
        """

        :param path:
        :param new_dir_name:

        """
        if os.path.exists(path):
            if not os.path.exists(f"{path}/{new_dir_name}"):
                os.makedirs(f"{path}/{new_dir_name}")
        else:
            raise Exception(f"Error, path={path} did not exist.")


if __name__ == "__main__":
    main = Plot_to_tex()
    main.example_create_a_table()
