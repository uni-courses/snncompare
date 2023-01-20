"""Exports the test results to a json file."""
import json
from pathlib import Path
from typing import Any, Dict, List, Union

import networkx as nx
from networkx.readwrite import json_graph
from typeguard import typechecked


@typechecked
def write_dict_to_json(output_filepath: str, some_dict: Dict) -> None:
    """Writes a dict file to a .json file."""

    with open(output_filepath, "w", encoding="utf-8") as fp:
        json.dump(some_dict, fp, indent=4, sort_keys=True)
        fp.close()

    # Verify the file exists.
    if not Path(output_filepath).is_file():
        raise Exception(f"Error, filepath:{output_filepath} was not created.")

    # TODO: verify the file content is valid.
    if "graph" in some_dict.keys():
        with open(output_filepath, encoding="utf-8") as json_file:
            graph_dict = json.load(json_file)
            json_file.close()
        some_graph = json_graph.node_link_graph(graph_dict)
        if not isinstance(some_graph, (nx.Graph, nx.DiGraph)):
            raise ImportError(
                f"Error, the json with a graph:{some_dict}, was not exported"
                + "to file successfully, because it was not loaded back into a"
                f"graph:{type(some_graph)}."
            )


@typechecked
def encode_tuples(some_dict: Dict, decode: bool = False) -> Dict:
    """Loops through the values of the dict and if it detects a list with
    tuples, it encodes the tuples for json exporting.

    If the decode parameter is True, then it decodes that encoded object
    back to tuple.
    """

    enc = MultiDimensionalArrayEncoder()
    for key, val in some_dict.items():
        if isinstance(val, tuple):
            val = enc.encode(val)
        elif isinstance(val, List):
            if key == "size_and_max_graphs":
                if decode:
                    some_dict[key] = list(
                        map(hinted_tuple_hook, map(json.loads, val))
                    )
            if all(isinstance(n, tuple) for n in val):
                some_dict[key] = list(map(enc.encode, val))
    return some_dict


class MultiDimensionalArrayEncoder(json.JSONEncoder):
    """Encodes tuples into a string such that they can be exported to a json
    file."""

    def encode(self, o: Any) -> Any:  # type:ignore[misc]
        def hint_tuples(  # type:ignore[misc]
            item: Union[tuple, List, Dict, Any]
        ) -> Any:
            if isinstance(item, tuple):
                return {"__tuple__": True, "items": item}
            if isinstance(item, list):
                return [hint_tuples(e) for e in item]
            if isinstance(item, Dict):
                return {key: hint_tuples(value) for key, value in item.items()}
            return item

        return super().encode(hint_tuples(o))


def hinted_tuple_hook(obj: Dict) -> Union[Dict, tuple]:
    """Checks if a dictionary contains the keyword __tuple__ and if yes,
    decodes it by returning the accompanying tuple stored in the value
    belonging to the "items" key."""
    if "__tuple__" in obj:
        return tuple(obj["items"])
    return obj
