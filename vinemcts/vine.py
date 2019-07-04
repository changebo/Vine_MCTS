import numpy as np
import igraph
from functools import reduce
import random


class VineState:
    def __init__(self, ntrunc, corr_mat):
        """ Constructor

        This function is only called when constructing the root state.
        Subsequent states are constructed by calling self._clone().

        Args:
            ntrunc: Number of truncation level.
            corr_mat: A CorrMat object.
        """
        self._ntrunc = ntrunc
        self._corr_mat = corr_mat

        # dimension
        self._d = corr_mat.dim()

        assert self._ntrunc > 0
        assert self._ntrunc < self._d

        # self.tree_list is a list of igraph objects, representing an
        # incomplete truncated vine. Each element is a tree, except for the
        # last one, which is an incomplete tree. When self.__init__() is
        # called, self.tree_list is a list with an empty igraph object.
        g = igraph.Graph()
        g.add_vertices(self._d)
        g.vs["name"] = [str(i) for i in range(self._d)]
        self.tree_list = [g]

        # The score of the incomplete vine: -log(1-r^2).
        self.score = 0.0

    def _clone(self):
        """ Create a deep clone of this state. """

        # _corr_mat is a shallow copy
        st = VineState(ntrunc=self._ntrunc, corr_mat=self._corr_mat)

        # Create a deep copy of self.tree_list
        st.tree_list = [g.copy() for g in self.tree_list]
        st.score = self.score

        return st

    def _level(self):
        """ Return the level of the current incomplete vine. 

        Level is zero-based. 
        """
        return len(self.tree_list) - 1

    def _is_complete(self):
        """ Whether a vine state is complete.

        A vine state is complete if the last tree in self.tree_list is a 
        connected tree, and the current level reaches the truncation level.
        """
        self_g = self.tree_list[-1]
        return (self_g.ecount() == self_g.vcount() - 1) and (
            self._level() >= self._ntrunc - 1
        )

    def get_child_states(self):
        """ Get a list of all valid child states.

        If there is none, return an empty list.
        """

        if self._is_complete():
            return []

        self_g = self.tree_list[-1]

        # Append an empty graph to self.tree_list if self_g is a connected
        # tree.
        if self_g.ecount() == self_g.vcount() - 1:
            # self_g is already a tree.
            # If the current tree is connected but it hasn't reached ntrunc,
            # then add another empty graph.
            g = igraph.Graph()
            g.add_vertices(self_g.ecount())
            g.vs["name"] = self_g.es["name"]
            self.tree_list.append(g)
            self_g = self.tree_list[-1]

        # Initialize the returned list.
        res = []

        if self_g.ecount() == 0:
            # If self_g is empty, select all pairs of edges as child states.

            for i in range(self_g.vcount()):
                for j in range(i):
                    # Connect i and j.
                    st = self._add_edge_helper(i, j)
                    if st is not None:
                        res.append(st)

        else:
            # If self_g is NOT empty, connect vertices with degree > 0 and
            # vertices with degree == 0. By doing so, there is always only one
            # connected component in the graph. The way we grow the tree
            # resembles Prim's algorithm, not Kruskal's algorithm.

            adj_mat = self_g.get_adjacency()
            adj_vec = [max(a) for a in adj_mat]

            # Vertices with degree > 0
            conn_ids = [i for i, x in enumerate(adj_vec) if x == 1]
            # Vertices with degree == 0
            disconn_ids = [i for i, x in enumerate(adj_vec) if x == 0]

            for i in conn_ids:
                for j in disconn_ids:
                    # Connect i and j.
                    st = self._add_edge_helper(i, j)
                    if st is not None:
                        res.append(st)

        return res

    def _add_edge_helper(self, i, j):
        """ Add an edge to the last graph in self.tree_list.

        Add an edge between vertex id i and j, if proximity condition is 
        satisfied. The score is updated.

        Args:
            i, j: vertex indices in self.tree_list[-1].
            res: a list which the result state is appended to.

        Returns:
            A VineState with the added edge.
            If proximity condition is not satisfied, return None.
        """

        # Create a deep copy of the current state.
        temp_st = self._clone()

        # copy_g is the last incomplete tree in the newly copied state.
        copy_g = temp_st.tree_list[-1]

        if self._level() == 0:
            # If there's only one tree in self.tree_list, no need to consider
            # the proximity condition. Simply add an edge.

            copy_g.add_edges([(i, j)])

            # Add edge name
            copy_g.es[copy_g.ecount() - 1]["name"] = ",".join(
                [str(j), str(i)] if j < i else [str(i), str(j)]
            )

            # Update score
            this_score = -np.log(1 - self._corr_mat.pcorr(i, j) ** 2)
            temp_st.score += this_score
            copy_g.es[copy_g.ecount() - 1]["weight"] = this_score

        else:
            # When level > 1, check the proximity condition first.
            # If it is not satisfied, return None.
            # Otherwise, add the edge.

            prev_g = temp_st.tree_list[-2]

            # Get vertex names of i, j in copy_g
            i_v_name = copy_g.vs[i]["name"]
            j_v_name = copy_g.vs[j]["name"]

            # Get edge ids in prev_g
            i_edge = prev_g.es.find(name=i_v_name)
            j_edge = prev_g.es.find(name=j_v_name)

            if not set(i_edge.tuple) & set(j_edge.tuple):
                # If the intersection of i_edge and j_edge is empty, then the
                # proximity condition is not satisfied. Skip this pair.
                return None

            # Proximity condition is satisfied.
            copy_g.add_edges([(i, j)])

            # Assertions
            if i_v_name.find("|") >= 0:
                assert j_v_name.find("|") >= 0
            elif i_v_name.find("|") < 0:
                assert j_v_name.find("|") < 0

            # Vertex names
            i_v_name_set = set(i_v_name.replace("|", ",").split(","))
            j_v_name_set = set(j_v_name.replace("|", ",").split(","))
            # Symmetric difference
            new_name_before_bar = ",".join(sorted(i_v_name_set ^ j_v_name_set))
            # Intersection
            new_name_after_bar = ",".join(sorted(i_v_name_set & j_v_name_set))
            new_name = new_name_before_bar + "|" + new_name_after_bar

            # Add edge name
            copy_g.es[copy_g.ecount() - 1]["name"] = new_name

            # Update score
            _i, _j = [int(k) for k in new_name_before_bar.split(",")]
            _S = [int(k) for k in new_name_after_bar.split(",")]
            this_score = -np.log(1 - self._corr_mat.pcorr(i=_i, j=_j, S=_S) ** 2)
            temp_st.score += this_score
            copy_g.es[copy_g.ecount() - 1]["weight"] = this_score

        return temp_st

    def roll_out(self):
        """ Roll out the current vine state to a complete one.
        The current implementation is naive. It randomly chooses a child state
        iteratively until reaching the end.

        Returns:
            (score, vine): The score and final vine state.
        """

        st = self._clone()

        while not st._is_complete():
            self_g = st.tree_list[-1]

            # Append an empty graph to self.tree_list if self_g is a connected
            # tree.
            if self_g.ecount() == self_g.vcount() - 1:
                # self_g is already a tree.
                # If the current tree is connected but it hasn't reached
                # ntrunc, then add another empty graph.
                g = igraph.Graph()
                g.add_vertices(self_g.ecount())
                g.vs["name"] = self_g.es["name"]
                st.tree_list.append(g)
                self_g = st.tree_list[-1]

            if self_g.ecount() == 0:
                # If self_g is empty, randomly pick a pair of edges as child
                # states.

                v_list = list(range(self_g.vcount()))
                random.shuffle(v_list)

                found = False
                for i in v_list:
                    for j in range(i):
                        # Connect i and j.
                        temp_st = st._add_edge_helper(i, j)
                        if temp_st is not None:
                            st = temp_st
                            found = True
                            break

                    if found:
                        break

            else:
                # If self_g is NOT empty, connect vertices with degree > 0 and
                # vertices with degree == 0. By doing so, there is always only
                # one connected component in the graph. The way we grow the
                # tree resembles Prim's algorithm, not Kruskal's algorithm.

                adj_mat = self_g.get_adjacency()
                adj_vec = [max(a) for a in adj_mat]

                # Vertices with degree > 0
                conn_ids = [i for i, x in enumerate(adj_vec) if x == 1]
                # Vertices with degree == 0
                disconn_ids = [i for i, x in enumerate(adj_vec) if x == 0]

                random.shuffle(conn_ids)
                random.shuffle(disconn_ids)

                found = False
                for i in conn_ids:
                    for j in disconn_ids:
                        # Connect i and j.
                        temp_st = st._add_edge_helper(i, j)
                        if temp_st is not None:
                            st = temp_st
                            found = True
                            break

                    if found:
                        break

        return (st.score, st)

    def to_vine_array(self):
        """ Convert an object to a vine array representation. 
        The representation is one-based.

        Return a d-by-d upper triagular matrix.
        """

        d = self._d

        # clone is a full vine, randomly rolled out from the current truncated
        # vine.
        clone = self._clone()
        clone._ntrunc = d - 1
        _, clone = clone.roll_out()

        # cond_sets is a list of length d-1,
        # each element is a list of conditioned sets at each level.
        cond_sets = []
        for k in range(d - 1):
            current_edges = clone._edge_repr()[
                k * d - (k ** 2 + k) // 2 : (k + 1) * d - ((k + 1) ** 2 + (k + 1)) // 2
            ]
            current_edges = [
                [int(node) for node in e.split("|")[0].split(",")]
                for e in current_edges
            ]
            # print(current_edges)
            cond_sets.append(current_edges)

        # When constructing the vine array, that elements are added column by
        # column, from right to left.
        # Within each column, elements are added from bottom to top.
        # In other words, we start from the last tree.
        M = -np.ones((d, d), dtype=np.int)
        for k in range(d - 2, -1, -1):
            w = cond_sets[k][0][0]
            M[k + 1, k + 1] = w
            M[k, k + 1] = cond_sets[k][0][1]
            del cond_sets[k][0]
            for ell in range(k - 1, -1, -1):
                for j in range(len(cond_sets[ell])):
                    if w in cond_sets[ell][j]:
                        cond_sets[ell][j].remove(w)
                        v = cond_sets[ell][j][0]
                        M[ell, k + 1] = v
                        del cond_sets[ell][j]
                        break

        M[0, 0] = M[0, 1]
        M += 1  # change from zero-based to one-based

        return M

    def _edge_repr(self):
        """ Edge representation of the VineState. 

        Returns a list of strings, each represents an edge. 
        For example: 
        ['0,2', '1,3', '2,4', '3,5', '4,6', '5,6', '2,6|4', '3,6|5', '4,5|6']
        """
        res = [sorted(g.es["name"]) for g in self.tree_list if g.ecount() > 0]
        if res:
            res = reduce(lambda x, y: x + y, res)

        return res

    def __hash__(self):
        return hash(tuple(self._edge_repr()))

    def __eq__(self, other):
        return self.__hash__() == other.__hash__()

    def __repr__(self):
        return self._edge_repr().__repr__()


class CorrMat:
    """ A wrapper of a correlation matrix. """

    def __init__(self, corr_mat, n):
        """ Constructor.

        Args:
            corr_mat: a correlation matrix as a numpy array.
        """

        # corr_mat should be a square matrix
        assert corr_mat.ndim == 2
        assert corr_mat.shape[0] == corr_mat.shape[1]

        self._corr_mat = corr_mat
        self._corr_mat_inv = np.linalg.inv(self._corr_mat)
        self._n = n

    def n_sample(self):
        return self._n

    def dim(self):
        """ Number of variables in the correlation matrix. """
        return self._corr_mat.shape[0]

    def log_det(self):
        """ Log determinant of the correlation matrix. """
        return np.log(np.linalg.det(self._corr_mat))

    def pcorr(self, i, j, S=None):
        """ Partial correlation of (i,j)|S. The indices are zero based.

        Args:
            i, j: Indices.
            S: A list of indices.
        """
        if not S:
            return self._corr_mat[i][j]

        ind = [i, j] + S
        sub_matrix = self._corr_mat[np.ix_(ind, ind)]

        # TODO: consider using np.linalg.solve instead of np.linalg.inv in the
        # future.
        sub_matrix_inv = np.linalg.inv(sub_matrix)
        return -sub_matrix_inv[0, 1] / np.sqrt(
            sub_matrix_inv[0, 0] * sub_matrix_inv[1, 1]
        )

    def _parse_edge_name(self, name):
        """ Parse the name of an edge. For example: name ='2,5|3,6'.

        Args:
            name: name of an edge. For example: name ='2,5|3,6'.

        Returns:
            i, j, S
        """
        name_split = name.split("|")
        i, j = [int(k) for k in name_split[0].split(",")]
        if len(name_split) > 1:
            S = [int(k) for k in name_split[1].split(",")]
        else:
            S = None

        return i, j, S

    def pcorr_by_name(self, name):
        """ Partial correlation by name. For example: name ='2,5|3,6'. """

        return self.pcorr(*self._parse_edge_name(name))

    def pcorr_given_all_by_name(self, name):
        """ Partial correlation of i, j given all other variables.

        Args:
            name: name of an edge. For example: name ='2,5|3,6'.

        Returns:
            Partial correlation of i, j given all other variables.
        """
        i, j, _ = self._parse_edge_name(name)
        return -self._corr_mat_inv[i, j] / np.sqrt(
            self._corr_mat_inv[i, i] * self._corr_mat_inv[j, j]
        )

    def __repr__(self):
        return self._corr_mat.__repr__()
