# -*- coding: utf-8 -*-
"""
Created on Fri Mar 23 17:08:21 2018

@author: Morgan.Li
"""

from collections import defaultdict
from heapq import heappop, heappush
import copy
import numpy as np


class Graph:
    '''
    Dijkstra shortest path algorithm based on python heapq heap implementation
    
    '''    
    def __init__(self):
        self.nodes = set()
        self.edges = defaultdict(list)
        self.distances = {}

    def add_node(self, value):
        self.nodes.add(value)

    def del_node(self, value):
        self.nodes.remove(value)
    
    def add_edge(self, from_node, to_node, distance, is_duplex=True):
        if from_node not in self.nodes:
            self.add_node(from_node)
        if to_node not in self.nodes:
            self.add_node(to_node)            
        self.edges[(from_node, to_node)] =  distance  

        if(is_duplex):
            self.edges[(to_node, from_node)] =  distance  
            
    def del_edge(self, from_node, to_node, is_duplex=True):
        self.edges.pop((from_node, to_node))
        if(is_duplex):
            self.edges.pop((to_node, from_node))

    def get_border_cost(self, node):
        border_cost = {}
        for key in self.edges:
            l,r = key
            if l == node:
                c = self.edges[key]
                border_cost[r] = c
        #sort dict based on value
        return sorted(border_cost.items(), key=lambda x: x[1])
    
    def get_cost_with_direct_borders(self):
        cost_and_node = {}
        for center_node in self.nodes:
            other_nodes = copy.deepcopy(self.nodes)
            other_nodes.remove(center_node)
            border_cost = []
            for other_node in other_nodes:
                c = self.edges[(other_node, center_node)]
                border_cost.append(c)         
            cost_and_node[center_node] = np.mean(border_cost)    
        node_list = sorted(cost_and_node, key=cost_and_node.get)     
        return node_list, cost_and_node         
    
    def get_cost_of_nodes(self):
        """
        get the best node which has lowest cost with others nodes
        """
        cost_and_node = {}
        for center_node in self.nodes:
            other_nodes = copy.deepcopy(self.nodes)
            other_nodes.remove(center_node)
            cost = 0
            for other_node in other_nodes:
                cost += self.dijkstra(other_node, center_node)[0] 
            cost_and_node[center_node] = cost   
        node_list = sorted(cost_and_node, key=cost_and_node.get)     
        return node_list, cost_and_node        
    
    def dijkstra(self, source, dest):
        g = defaultdict(list)
        for key in self.edges:
            l,r = key
            c = self.edges[key]
            g[l].append((c,r))
    
        q, seen = [(0,source,())], set()
        while q:
            (cost,v1,path) = heappop(q)
            path = list(path)
            if v1 not in seen:
                seen.add(v1)    
                path.append(v1)
                if v1 == dest: return (cost, path)
    
                for c, v2 in g.get(v1, ()):
                    if v2 not in seen:
                        heappush(q, (cost+c, v2, path))
    
        return float("inf")
'''
@test code 
if __name__ == "__main__":
    edges = [
        ("A", "B", 17),
        ("A", "D", 5),
        ("B", "C", 8),
        ("B", "D", 9),
        ("B", "E", 7),
        ("C", "E", 5),
        ("D", "E", 15),
        ("D", "F", 6),
        ("E", "F", 8),
        ("E", "G", 9),
        ("F", "G", 11),
        ("B", "A", 7)
    ]
    graph = Graph()
    for i in edges:
        graph.add_edge(i[0], i[1], i[2])
    print ("=== Dijkstra ===")
    print (graph.edges)
    print ("A -> B:")
    print (graph.dijkstra("A", "B"))
    print ("A -> E:")
    print (graph.dijkstra("A", "E"))
    print ("F -> G:")   #output: (14, ('E', ('B', ('A', ()))))
    print (graph.dijkstra("F", "G"))
'''