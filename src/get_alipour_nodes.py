import networkx as nx


def get_alipour_nodes(
    G,
    m,
    rand_props,
):
    delta = rand_props.delta
    inhibition = rand_props.inhibition
    rand_ceil = rand_props.rand_ceil
    rand_nrs = rand_props.rand_nrs

    # Reverse engineer uninhibited spread rand nrs:
    # TODO: read out from rand_props object.
    uninhibited_spread_rand_nrs = [(x + inhibition) for x in rand_nrs]

    for node in G.nodes:
        # Initialise values.
        # G.nodes[node]["marks"] = 0
        G.nodes[node]["marks"] = G.degree(node) * (rand_ceil + 1) * delta
        G.nodes[node]["countermarks"] = 0
        G.nodes[node]["random_number"] = 1 * uninhibited_spread_rand_nrs[node]
        G.nodes[node]["weight"] = (
            G.degree(node) * (rand_ceil + 1) * delta
            + G.nodes[node]["random_number"]
        )
        G.nodes[node]["inhibited_weight"] = (
            G.nodes[node]["weight"] - inhibition
        )
    # Compute the mark based on degree+randomness=weight
    for node in G.nodes:
        max_weight = max(
            G.nodes[n]["weight"] for n in nx.all_neighbors(G, node)
        )

        nr_of_max_weights = 0
        for n in nx.all_neighbors(G, node):
            if (
                G.nodes[n]["weight"] == max_weight
            ):  # should all max weight neurons be marked or only one of them?

                # Always raise mark always by (rand_ceil + 1) * delta (not by 1).
                # Read of the score from countermarks, not marks.
                G.nodes[n]["marks"] += (rand_ceil + 1) * delta
                G.nodes[n]["countermarks"] += 1
                nr_of_max_weights = nr_of_max_weights + 1

                # Verify there is only one max weight neuron.
                if nr_of_max_weights > 1:
                    raise Exception("Two numbers with identical max weight.")

    # Don't compute for m=0
    for _ in range(1, m + 1):
        for node in G.nodes:
            G.nodes[node]["weight"] = (
                G.nodes[node]["marks"] + G.nodes[node]["random_number"]
            )
            G.nodes[node]["inhibited_weight"] = (
                G.nodes[node]["weight"] - inhibition
            )
            # Reset marks.
            G.nodes[node]["marks"] = 0
            G.nodes[node]["countermarks"] = 0

        for node in G.nodes:
            max_weight = max(
                G.nodes[n]["weight"] for n in nx.all_neighbors(G, node)
            )
            for n in nx.all_neighbors(G, node):
                if G.nodes[n]["weight"] == max_weight:

                    # Always raise mark always by (rand_ceil + 1) * delta (not by 1).
                    G.nodes[n]["marks"] += (rand_ceil + 1) * delta
                    G.nodes[n]["countermarks"] += 1

    counter_marks = []
    for node in G.nodes:
        counter_marks.append(G.nodes[node]["countermarks"])
        print(f'node:{node}, ali-mark:{G.nodes[node]["countermarks"]}')
    return counter_marks
