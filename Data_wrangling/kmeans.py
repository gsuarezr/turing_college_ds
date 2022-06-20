from typing import Iterable,Tuple,Sequence,Dict,List
from collections import defaultdict
from math import fsum,sqrt
from pprint import pprint
from random import sample
from functools import partial

Point=Tuple[int,...]
Centroid=Point

def mean(data:Iterable[float])->float:
    'Accurate Aritmethic mean'
    data=list(data)
    return fsum(data)/len(data)

def dist(p:Point,q:Point,fsum=fsum,sqrt=sqrt,zip=zip)->float:
    'Euclidean distance'
    return sqrt(fsum([(x-y)**2 for x,y in zip(p,q)]))

def assign_data(centroids:Sequence[Centroid],data:Iterable[Point])->Dict[Centroid,List[Point]]:
    'Group the data points to the closest centroid'
    d=defaultdict(list)
    for point in data:
        closest_centroid=min(centroids,key=partial(dist,point))
        d[closest_centroid].append(point)
    return dict(d)

def compute_centroid(groups:Iterable[Sequence[Point]])->List[Centroid]:
    'Compute centroid of each group'
    return [tuple(map(mean,zip(*group))) for group in groups]

def k_means(data,k=2,iterations=500):
    """
    Performs the K-means algorithm, k denotes the number of centroids(clusters), and the data must be passed as a list of tuples 
    """
    data=list(data)
    centroids=sample(data,k)
    for i in range(iterations):
        labeld=assign_data(centroids,data)
        centroids=compute_centroid(labeld.values())
    return centroids,labeld # At some point it would be nice to find out why this is not stable