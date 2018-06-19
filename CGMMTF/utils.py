# TODO move this function OUT!
def compute_statistics(self, adjacency_lists, last_states, prev_statistics=None):
    """
    :param last_states: the last array of states
    :param prev_statistics: the statistics needed: list of numpy matrices UxAxC2
    :return: the statistics needed for this model, according to the Lprec parameter
    """

    # Compute statistics
    new_statistics = np.zeros((len(adjacency_lists), self.A, self.C2))

    for u in range(0, len(adjacency_lists)):
        incident_nodes = adjacency_lists[u]
        for u2, a in incident_nodes:
            node_state = last_states[u2]

            # THIS CHECK IS USEFUL FOR COLLECTIVE INFERENCE
            if node_state != -1:
                new_statistics[u, a, node_state] += 1

    # Save it into a field for future use (incremental inference/training) --> strongly coupled
    last_statistics = np.reshape(new_statistics, (len(adjacency_lists), 1, self.A, self.C2))

    # take just the stats I am interested in
    if prev_statistics is not None:

        # Todo this is responsible for the increase in computation time while
        all_statistics = np.concatenate([np.reshape(last_statistics, (len(adjacency_lists), 1, self.A, self.C2)),
                                         prev_statistics], axis=1)  # UxLxAxC2

        bound = -1

        if all_statistics.shape[1] >= self.Lprec[bound]:  # I can take all the desired precedent layers
            all_statistics = all_statistics[:, self.Lprec - 1, :, :]

        else:  # take the max previous number of states

            while all_statistics.shape[1] < self.Lprec[bound]:  # Lprec[bound] is still too much
                bound -= 1

            all_statistics = all_statistics[:, self.Lprec[:bound + 1] - 1, :, :]

            # L has to adapt to the number of preceeding layers
            self.L = len(self.Lprec[:bound + 1])

    else:
        # Update parameters dimension (during training parameters will be initialised)
        self.L = 1
        self.Lprec = np.array([0])
        all_statistics = np.reshape(new_statistics, (len(adjacency_lists), 1, self.A, self.C2))

    # print("L is ", self.L)

    return all_statistics
