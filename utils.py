import numpy as np
import torch
import networkx as nx
from scipy.spatial import Delaunay
from scipy.spatial import Voronoi
from scipy.spatial import cKDTree


def ccw(A,B,C):
    return (C[1]-A[1]) * (B[0]-A[0]) > (B[1]-A[1]) * (C[0]-A[0])


def intersect(A,B,C,D):
    return ccw(A,C,D) != ccw(B,C,D) and ccw(A,B,C) != ccw(A,B,D)


def DelaunayGraph(points):
    tri = Delaunay(points)
    G = nx.Graph()

    indptr = tri.vertex_neighbor_vertices[0]
    indices = tri.vertex_neighbor_vertices[1]
    for i in range(len(points)):
        for j in indices[indptr[i]:indptr[i + 1]]:
            if i < j:
                G.add_edge(i, j)

    return G


def GabrielViaDelaunayVoronoi(points):

    delaunayGraph = DelaunayGraph(points)
    voronoiDiagram = Voronoi(points)
    voronoiVertices = voronoiDiagram.vertices
    voronoiCenter = voronoiDiagram.points.mean(axis=0)
    ptp_bound = voronoiDiagram.points.ptp(axis=0)

    gabrielGraph = nx.Graph()
    for u, v in delaunayGraph.edges():
        uRegion = set(voronoiDiagram.regions[voronoiDiagram.point_region[u]])
        vRegion = set(voronoiDiagram.regions[voronoiDiagram.point_region[v]])
        boundary = sorted(list(uRegion.intersection(vRegion)))[-2:]
        boundaryVertices = [None, voronoiVertices[boundary[1]]]

        if (boundary[0] == -1):
            tangent = points[u] - points[v]
            tangent /= np.linalg.norm(tangent)
            normal = np.array([-tangent[1], tangent[0]])

            midPoint = 0.5 * (points[u] + points[v])
            direction = np.sign(np.dot(midPoint - voronoiCenter, normal)) * normal
            farPoint = voronoiVertices[boundary[1]] + direction * ptp_bound.max()
            boundaryVertices[0] = farPoint
        else:
            boundaryVertices[0] = voronoiVertices[boundary[0]]

        if intersect(points[u], points[v], boundaryVertices[0], boundaryVertices[1]):
            gabrielGraph.add_edge(u, v)

    return gabrielGraph


def EpsilonGraph(points, epsilon):
    tree = cKDTree(points)
    epsilonGraph = nx.Graph()
    for i, p in enumerate(points):
        neighbors = tree.query_ball_point(p, epsilon)
        for j in neighbors:
            if j > i:  # avoid duplicates & self
                d = torch.sqrt(torch.sum((points[i] - points[j]) ** 2)).detach().cpu().item()
                epsilonGraph.add_edge(i, j, weight=1/d)
    return epsilonGraph

def clean_graph(graph):

    H = graph.copy()  # work on a clone
    # longest-first list of (u, v, data) triples
    edges = sorted(H.edges(data=True),
                   key=lambda e: e[2].get("weight", 1),
                   reverse=True)

    for u, v, data in edges:
        H.remove_edge(u, v)  # tentatively drop it
        if not nx.is_connected(H):  # needed for connectivity?
            H.add_edge(u, v, **data)

    return H


























