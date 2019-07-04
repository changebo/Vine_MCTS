import math
import random
import numpy as np
from vinemcts.vine import *


class MctsNode:
    """ A node class of the search tree. """

    def __init__(self, config, state=None):
        """ Constructor

        Args:
            config: A configuration dictionary, contains
                transpos_table, UCT_const, FPU.
        """

        self.state = state
        self.config = config
        self.child_nodes = []
        # self.child_visits keeps how many times the child nodes are visitied
        # from the *current* node.
        self.child_visits = []
        self.visits = 0
        self.sum_score = 0

    def select_child(self):
        """ Tree policy: Select a child from the transposition table according 
        to UCT.

        Update self.child_visits.
        Note: self.visits is not updated here. It is updated when self.update 
        is called.
        """

        # UCT_score uses UCT2 in Childs et al. (2008). Transpositions and move
        # groups in monte carlo tree search.
        def UCT(c_node, c_node_visits):
            mean_score = (c_node.sum_score / c_node.visits) if c_node.visits > 0 else 0

            # Margin of Error
            if c_node_visits == 0:
                moe = self.config["FPU"]
            else:
                moe = math.sqrt(2 * math.log(self.visits + 1) / c_node_visits)

            edge_score = c_node.state.score - self.state.score
            prog_bias = self.config["PB"] * edge_score / (c_node.visits + 1)

            # # An alternative progressive bias term:
            # # the partial correlation of the newly added edge, given all the
            # # other variables.
            # edge_diff_set = set(c_node.state._edge_repr()) - \
            #     set(self.state._edge_repr())
            # assert len(edge_diff_set) == 1
            # edge_diff = list(edge_diff_set)[0]
            # pcorr_all = self.state._corr_mat.pcorr_given_all_by_name(
            #     edge_diff)
            # prog_bias = self.config[
            #     'PB'] * (-np.log(1 - pcorr_all**2)) / (c_node.visits + 1)

            return mean_score + self.config["UCT_const"] * (moe + prog_bias)

        UCT_list = [
            (UCT(node, self.child_visits[i]), node.state, i)
            for i, node in enumerate(self.child_nodes)
        ]

        # max() in python 3: If multiple items are maximal, the function
        # returns the first one encountered.
        # Shuffle score_list so that when two nodes have the same UCT_score,
        # one of them is randomly picked.
        random.shuffle(UCT_list)

        max_UCT_list = max(UCT_list, key=lambda x: x[0])
        # print(max_UCT_list)
        selected_state = max_UCT_list[1]
        select_index = max_UCT_list[2]
        selected_node = self.config["transpos_table"][selected_state]

        # Update number of visits
        self.child_visits[select_index] += 1

        return selected_node

    def add_children(self):
        """ Add all children of the current node if possible.

        The children are added to the transposition table.
        Add the child *nodes* to self.child_nodes.
        Initialize self.child_visits to zeros.

        Returns:
            bool: If children are successfully added, return True.
                If not, do nothing and return False.

        """

        assert len(self.child_nodes) == 0

        child_states = self.state.get_child_states()

        if not child_states:
            return False

        self.child_visits = [0] * len(child_states)
        for c_state in child_states:
            if c_state not in self.config["transpos_table"]:
                self.config["transpos_table"][c_state] = MctsNode(
                    config=self.config, state=c_state
                )

            self.child_nodes.append(self.config["transpos_table"][c_state])

        self.child_visits = [0] * len(child_states)

        return True

    def roll_out(self):
        """ Run default policy

        Returns:
            (score, vine): The score and final vine state.

        """
        return self.state.roll_out()

    def update(self, result):
        """ Update the node with result """

        self.visits += 1
        self.sum_score += result

    def is_leaf(self):
        """ If the node is a leaf node."""

        # The node is leaf node if its self.child_nodes is empty.
        return self.child_nodes == []

    def __repr__(self):
        mean_score = str(self.sum_score / self.visits) if self.visits > 0 else "N/A"
        return "Vine state: " + self.state.__repr__() + "\nMean score: " + mean_score


def mcts_vine(
    corr, n_sample, ntrunc, output_dir, itermax=100, FPU=1.0, PB=0.1, log_freq=100
):
    # Initialize the correlation matrix object, root state and root node
    corr_mat = CorrMat(corr, n_sample)
    root_state = VineState(ntrunc=ntrunc, corr_mat=corr_mat)

    transpos_table = {}
    config = {
        # a dictionary: state -> node.
        "transpos_table": transpos_table,
        # UCB1 formula: \bar{x} + UCT_const \sqrt{log(n)/log(n_j)}
        "UCT_const": (-corr_mat.log_det()),
        # First Play Urgency
        "FPU": FPU,
        # Progressive Bias
        "PB": PB,
    }

    print("Configuration dictionary:")
    print(config)

    root_node = MctsNode(config, root_state)

    best_score = 0  # we want to maximize the score
    best_vine = None

    file_handler = open(output_dir, "w")

    # CFI calculation
    D_0 = corr_mat.n_sample() * (-corr_mat.log_det())
    nu_0 = corr_mat.dim() * (corr_mat.dim() - 1) / 2.0

    for i in range(itermax):
        node = root_node
        temp_node_list = [node]

        # Select
        while not node.is_leaf():
            node = node.select_child()
            temp_node_list.append(node)

        # Expand
        if node.visits > 0:
            # Only expand the leaf node if it has been visited.

            add_children_success = node.add_children()

            if add_children_success:
                node = node.select_child()
                temp_node_list.append(node)

        # Rollout
        score, vine = node.roll_out()

        if score > best_score:
            best_score = score
            best_vine = vine

            # CFI calculation
            D_ell = corr_mat.n_sample() * (-corr_mat.log_det() - best_score)
            nu_ell = (corr_mat.dim() - ntrunc) * (corr_mat.dim() - ntrunc - 1) / 2.0
            CFI = 1 - max(0, D_ell - nu_ell) / max(0, D_0 - nu_0, D_ell - nu_ell)

            file_handler.write("%d, %.4f, %.4f\n" % (i, best_score, CFI))

        if i % log_freq == 0 and i > 0:
            print(output_dir + ", Iter %d: " % i)
            print("best_score: " + str(best_score))

        # Backpropagate
        [node.update(score) for node in temp_node_list]

    # print("\nfinal vine, level %d: " % ntrunc)
    # d = corr_mat.dim()

    # for k in range(ntrunc):
    #     current_edges = best_vine._edge_repr()[
    #         k * d -
    #         (k**2 + k) // 2:(k + 1) * d -
    #         ((k + 1)**2 + (k + 1)) // 2]
    #     current_edges = [e.replace('|',
    #                                ',').split(',') for e in current_edges]
    #     for j in range(len(current_edges[0])):
    #         print([int(e[j]) + 1 for e in current_edges])

    # print("pcorr: ")
    # print([round(corr_mat.pcorr_by_name(s), 4)
    #        for s in best_vine._edge_repr()])

    print("CFI: " + str(CFI))
    file_handler.write("\nCFI: " + str(CFI) + "\n\n")

    best_vine_array = best_vine.to_vine_array()
    best_vine_array = best_vine_array.flatten()
    best_vine_array = np.array2string(best_vine_array, separator=",")
    file_handler.write(best_vine_array)

    file_handler.close()

    return best_vine


if __name__ == "__main__":
    pass
