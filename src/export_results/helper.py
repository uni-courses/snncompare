"""Contains helper functions for exporting simulation results."""
import collections
import copy


def flatten(d, parent_key="", sep="_"):
    """Flattens a dictionary (makes multiple lines into a oneliner)."""
    items = []
    for k, v in d.items():
        new_key = parent_key + sep + k if parent_key else k
        if isinstance(v, collections.MutableMapping):
            items.extend(flatten(v, new_key, sep=sep).items())
        else:
            items.append((new_key, v))
    return dict(items)


# >>> flatten({'a': 1, 'c': {'a': 2, 'b': {'x': 5, 'y' : 10}}, 'd': [1, 2, 3]})
# {'a': 1, 'c_a': 2, 'c_b_x': 5, 'd': [1, 2, 3], 'c_b_y': 10}


def run_config_to_filename(run_config: dict) -> str:
    """Converts a run_config dictionary into a filename.

    Does that by flattining the dictionary (and all its child-
    dictionaries).
    """
    # TODO: order dictionaries by alphabetical order by default.
    # TODO: allow user to specify a custom order of parameters.

    stripped_run_config = copy.deepcopy(run_config)
    stripped_run_config.pop("unique_id")  # Unique Id will be added as tag
    # instead (To reduce filename length).
    filename = str(flatten(stripped_run_config))

    # Remove the ' symbols.
    # Don't, that makes it more difficult to load the dict again.
    # filename=filename.replace("'","")

    # Don't, that makes it more difficult to load the dict again.
    # Remove the spaces.
    filename = filename.replace(" ", "")

    if len(filename) > 256:
        raise Exception(f"Filename={filename} is too long:{len(filename)}")
    return filename
