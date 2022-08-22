import numpy as np
from typing import List
import random
import math
import copy
import matplotlib.pyplot as plt
import time

from python_tsp.heuristics import solve_tsp_local_search


#START VERTEX IS NOT IN "VERTICES (vertices)"
global dist_calls
dist_calls = 0
class Vertex():

    def __init__(self, name, x, y, visit_duration, time_window, star_rating):
        self.name = name
        self.x = x
        self.y = y
        self.visit_duration = visit_duration
        self.time_window = time_window
        self.utility = star_rating*self.visit_duration
    
    def distance_to(self, other_vertex):
        global dist_calls
        dist_calls += 1
        other_x, other_y = other_vertex.x, other_vertex.y
        if other_x > self.x:
            return (np.sqrt((self.x - other_x)**2 + (self.y - other_y)**2))
        else:
            return (np.sqrt((self.x - other_x)**2 + (self.y - other_y)**2))
    
    def __str__(self):
        return self.name

def utility(path, taus=None):
    utility = 0
    #TODO: time windows accounting? Necessary? Double check paper section on this
    for vertex in path:
        utility += vertex.utility
    return utility

def cost(path):
    cost = 0
    for i in range(1, len(path)):
        cost += path[i - 1].distance_to(path[i])
        cost += path[i].visit_duration
    cost += path[-1].distance_to(path[0])
    return cost

global tsp_time
tsp_time = 0
def tsp(vertices):
    global tsp_time
    tsp_time -= time.time()
    #TODO: Is this the right way to do distance matrix?
    distance_matrix = np.asarray([[vertex1.distance_to(vertex2) for vertex2 in vertices] for vertex1 in vertices])
    permutation, distance = solve_tsp_local_search(distance_matrix)

    """
    tour = random.sample(range(len(vertices)),len(vertices))
    for temperature in np.logspace(0,3,num=1000)[::-1]:
        [i,j] = sorted(random.sample(range(len(vertices)),2))
        newTour =  tour[:i] + tour[j:j+1] +  tour[i+1:j] + tour[i:i+1] + tour[j+1:]
        oldDistances =  sum([ vertices[tour[k % len(vertices)]].distance_to(vertices[tour[(k + 1) % len(vertices)]]) for k in [j,j-1,i,i-1]])
        newDistances =  sum([ vertices[newTour[k % len(vertices)]].distance_to(vertices[newTour[(k + 1) % len(vertices)]]) for k in [j,j-1,i,i-1]])
        if math.exp( ( oldDistances - newDistances) / temperature) > random.random():
            tour = copy.copy(newTour)
    """
    tsp_time += time.time()
    return [vertices[index] for index in permutation] #in tour]

global greedy_tsp_time
greedy_tsp_time = 0
def greedy_tsp_cost(vertices: List[Vertex], budget: int, start_vertex: Vertex):
    global greedy_tsp_time
    greedy_tsp_time -= time.time()
    vertices = copy.copy(vertices)
    vertices = set(vertices)
    Ps = [start_vertex]
    while Ps is not None:
        P = Ps
        best_margin = 0
        Ps = None
        added_vertex = None
        for vertex in vertices:
            tsp_sol = tsp(P + [vertex])
            Pp = list(np.roll(tsp_sol, -1*tsp_sol.index(start_vertex)))
            assert type(Pp) == type([]), type(Pp)
            assert Pp[0] == start_vertex
            margin = (utility(Pp) - utility(P))/(cost(Pp) - cost(P))
            if margin > best_margin and cost(Pp) < budget:
                best_margin = margin
                Ps = Pp
                added_vertex = vertex
        #afterwards
        if added_vertex is not None:
            vertices.remove(added_vertex)
            #for vertices in Ps but not in P, remove from vertices
    tauP = [0]*len(P)
    for i in range(1, len(P)):
        tauP[i] = tauP[i - 1] + P[i - 1].visit_duration + P[i - 1].distance_to(P[i])
    greedy_tsp_time += time.time()
    return (P, tauP)

def truncate_tour(P, tauP, target):
    P = copy.copy(P)[1:]
    start_vertex = P[0]
    tauP = copy.copy(tauP)[1:]
    for tau, vertex in zip(tauP, P):
        P.remove(vertex)
        tauP.remove(tau)
        if utility([start_vertex] + P) <= 2*target:
            break
    return ([start_vertex] + P, [0] + tauP)

def make_two_node_tour(start_vertex, other_vertex):
    return ([start_vertex, other_vertex], [0, start_vertex.distance_to(other_vertex)])

def max_min_orienteering(vertices: List[Vertex], budget: int, number_of_required_tours: int, start_vertex: Vertex, target: int):
    vertices = copy.copy(vertices)
    multiday_tours = []
    multiday_taus = []
    remove_vertices = []
    for vertex in vertices:
        if vertex.utility < target:
            continue
        P, tauP = make_two_node_tour(start_vertex, vertex)
        multiday_tours.append(P)
        multiday_taus.append(tauP)
        remove_vertices.append(vertex)
    [vertices.remove(vertex) for vertex in remove_vertices]
    if len(multiday_tours) >= number_of_required_tours:
        return (multiday_tours, multiday_taus)
    
    while len(multiday_tours) < number_of_required_tours:
        P, tauP = greedy_tsp_cost(vertices=vertices, budget=budget, start_vertex=start_vertex)
        P, tauP = truncate_tour(P, tauP, target)
        multiday_tours.append(P)
        multiday_taus.append(tauP)
        [vertices.remove(vertex) for vertex in P[1:]]
    
    for P, tauP in zip(multiday_tours, multiday_taus):
        if utility(path=P, taus=tauP) < target:
            return (None, None)
    return (multiday_tours, multiday_taus)

vertices = []
for i in range(500):
    vertex = Vertex(
        name=str(i),
        x=random.random()*100,
        y = random.random()*100,
        visit_duration=random.random()*30 + 15,
        time_window = [0, 480],
        star_rating=random.random()*4 + 1, #float between 1 and 5
    )
    vertices.append(vertex)

budget = 8*60
number_of_required_tours = 1
start_vertex = Vertex(name='start', x=20, y=20, visit_duration=0, time_window=[0, 0], star_rating=0)
target = 120*4
print("Starting alg")
overall_time = -time.time()
multiday_tours, multiday_taus = max_min_orienteering(vertices=vertices, budget=budget, number_of_required_tours=number_of_required_tours, start_vertex=start_vertex, target=target)
overall_time += time.time()

print(overall_time)
print(greedy_tsp_time)
print(tsp_time)

print(dist_calls)


v_xs = [vertex.x for vertex in vertices]
v_ys = [vertex.y for vertex in vertices]
v_utilities = [vertex.utility for vertex in vertices]
plt.scatter(v_xs, v_ys, c=v_utilities)
plt.gray()
for tour in multiday_tours:
    x = [vertex.x for vertex in tour] + [start_vertex.x]
    y = [vertex.y for vertex in tour] + [start_vertex.y]
    plt.plot(x, y, 'xb-')

plt.scatter([start_vertex.x], [start_vertex.y], c='red')

plt.show()
