from Graph import Graph
from Search import Search
from Pacman import Visualize


graph = Graph.readfile("task02_pacman_example_map.txt")
search = Search(graph, graph.start)
path, cost = search.astar_search()
visualize = Visualize(graph, path)
visualize.path_Pacman()