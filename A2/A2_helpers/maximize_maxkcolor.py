""" Classes for defining fitness functions."""

# Author: Genevieve Hayes (Modified by Andrew Rollings)
# License: BSD 3 clause
from mlrose_hiive.fitness import MaxKColor

import numpy as np
##ALL from MLROS_HIIVE

# class MaxKColorCustomMaximize(MaxKColor):
#     """Fitness function for Max-k color optimization problem. Evaluates the
#     fitness of an n-dimensional state vector
#     :math:`x = [x_{0}, x_{1}, \\ldots, x_{n-1}]`, where :math:`x_{i}`
#     represents the color of node i, as the number of pairs of adjacent nodes
#     of the same color.

#     Parameters
#     ----------
#     edges: list of pairs
#         List of all pairs of connected nodes. Order does not matter, so (a, b)
#         and (b, a) are considered to be the same.

#     Example
#     -------
#     .. highlight:: python
#     .. code-block:: python

#         >>> import mlrose_hiive
#         >>> import numpy as np
#         >>> edges = [(0, 1), (0, 2), (0, 4), (1, 3), (2, 0), (2, 3), (3, 4)]
#         >>> fitness = mlrose_hiive.MaxKColor(edges)
#         >>> state = np.array([0, 1, 0, 1, 1])
#         >>> fitness.evaluate(state)
#         3

#     Note
#     ----
#     The MaxKColor fitness function is suitable for use in discrete-state
#     optimization problems *only*.

#     This is a cost minimization problem: lower scores are better than
#     higher scores. That is, for a given graph, and a given number of colors,
#     the challenge is to assign a color to each node in the graph such that
#     the number of pairs of adjacent nodes of the same color is minimized.
#     """

#     def __init__(self, edges):
#         super().__init__(edges)
#         self.maximize = True

#     def evaluate(self, state):
#         """Evaluate the fitness of a state vector.

#         Parameters
#         ----------
#         state: array
#             State array for evaluation.

#         Returns
#         -------
#         fitness: float
#             Value of fitness function.
#         """

#         fitness = 0

#         # this is the count of neigbor nodes with the same state value.
#         # Therefore state value represents color.
#         # This is NOT what the docs above say.

#         edges = self.edges if self.graph_edges is None else self.graph_edges
#         fitness = sum(int(state[n1] != state[n2]) for (n1, n2) in edges)
#         """
#         if fitness == 0:
#             for i in range(len(edges)):
#                 # Check for adjacent nodes of the same color
#                 n1, n2 = edges[i]
#                 print(f'i:{i}: ({n1},{n2})[{state[n1]}] <-> [{state[n2]}]')
#         """
#         """
        
#         if self.graph_edges is not None:
#             fitness = sum(int(state[n1] == state[n2]) for (n1, n2) in self.graph_edges)
#         else:
#             fitness = 0
#             for i in range(len(self.edges)):
#                 # Check for adjacent nodes of the same color
#                 if state[self.edges[i][0]] == state[self.edges[i][1]]:
#                     fitness += 1
#         """
#         return fitness
    
    
#     def set_graph(self, graph):
#         self.graph_edges = [e for e in graph.edges()]


# License: BSD 3 clause

import numpy as np

from mlrose_hiive.algorithms.crossovers import UniformCrossOver
from mlrose_hiive.algorithms.mutators import ChangeOneMutator
from mlrose_hiive.fitness import MaxKColor
from mlrose_hiive.opt_probs.discrete_opt import DiscreteOpt

import networkx as nx

    
    
    
from mlrose_hiive.opt_probs import MaxKColorOpt
import numpy as np
import networkx as nx


# class MaxKColorGeneratorCustom:
#     @staticmethod
#     def generate(seed, number_of_nodes=20, max_connections_per_node=4, max_colors=None):

#         """
#         >>> edges = [(0, 1), (0, 2), (0, 4), (1, 3), (2, 0), (2, 3), (3, 4)]
#         >>> fitness = mlrose_hiive.MaxKColor(edges)
#         >>> state = np.array([0, 1, 0, 1, 1])
#         >>> fitness.evaluate(state)
#         """
#         np.random.seed(seed)
#         # all nodes have to be connected, somehow.
#         node_connection_counts = 1 + np.random.randint(max_connections_per_node, size=number_of_nodes)

#         node_connections = {}
#         nodes = range(number_of_nodes)
#         for n in nodes:
#             all_other_valid_nodes = [o for o in nodes if (o != n and (o not in node_connections or
#                                                                       n not in node_connections[o]))]
#             count = min(node_connection_counts[n], len(all_other_valid_nodes))
#             other_nodes = sorted(np.random.choice(all_other_valid_nodes, count, replace=False))
#             node_connections[n] = [(n, o) for o in other_nodes]

#         # check connectivity
#         g = nx.Graph()
#         g.add_edges_from([x for y in node_connections.values() for x in y])

#         for n in nodes:
#             cannot_reach = [(n, o) if n < o else (o, n) for o in nodes if o not in nx.bfs_tree(g, n).nodes()]
#             for s, f in cannot_reach:
#                 g.add_edge(s, f)
#                 check_reach = len([(n, o) if n < o else (o, n) for o in nodes if o not in nx.bfs_tree(g, n).nodes()])
#                 if check_reach == 0:
#                     break

#         edges = [(s, f) for (s, f) in g.edges()]
#         problem = MaxKColorOpt(edges=edges, length=number_of_nodes, fitness_fn=MaxKColorCustomMaximize, maximize=True,
#                  max_colors=max_colors,source_graph=g)
#         return problem
    
    
    
    
class MaxKColorCustomMaximize(MaxKColor):
    """Custom Fitness function for Max-k color optimization problem to maximize the number of adjacent nodes with different colors."""

    def __init__(self, edges):
        super().__init__(edges)

    def evaluate(self, state):
        """Evaluate the fitness of a state vector.

        Parameters
        ----------
        state: array
            State array for evaluation.

        Returns
        -------
        fitness: float
            Value of fitness function.
        """
        edges = self.edges if self.graph_edges is None else self.graph_edges
        fitness = sum(int(state[n1] != state[n2]) for (n1, n2) in edges)
        return -1* fitness
    
    def set_graph(self, graph):
        self.graph_edges = [e for e in graph.edges()]

class MaxKColorGeneratorCustom:
    @staticmethod
    def generate(seed, number_of_nodes=20, max_connections_per_node=4, max_colors=None):
        np.random.seed(seed)
        node_connection_counts = 1 + np.random.randint(max_connections_per_node, size=number_of_nodes)

        node_connections = {}
        nodes = range(number_of_nodes)
        for n in nodes:
            all_other_valid_nodes = [o for o in nodes if (o != n and (o not in node_connections or n not in node_connections[o]))]
            count = min(node_connection_counts[n], len(all_other_valid_nodes))
            other_nodes = sorted(np.random.choice(all_other_valid_nodes, count, replace=False))
            node_connections[n] = [(n, o) for o in other_nodes]

        g = nx.Graph()
        g.add_edges_from([x for y in node_connections.values() for x in y])

        for n in nodes:
            cannot_reach = [(n, o) if n < o else (o, n) for o in nodes if o not in nx.bfs_tree(g, n).nodes()]
            for s, f in cannot_reach:
                g.add_edge(s, f)
                check_reach = len([(n, o) if n < o else (o, n) for o in nodes if o not in nx.bfs_tree(g, n).nodes()])
                if check_reach == 0:
                    break

        edges = [(s, f) for (s, f) in g.edges()]
        problem = MaxKColorOpt(edges=edges, length=number_of_nodes, fitness_fn=MaxKColorCustomMaximize(edges=edges), maximize=False, max_colors=max_colors, source_graph=g)
        return problem