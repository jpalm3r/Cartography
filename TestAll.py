#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu Nov  9 18:04:24 2017
@author: jaumep
"""

import os
from enredados1O import cartography

def gdf_to_nxdigraph(hashtag):
    input_gdf='../gdf_gephi/'+hashtag+'.gdf'

    import networkx as nx
    G = nx.DiGraph()

    # Open input file
    gdf_file = open(input_gdf,"r")

    # Deduce node attributes from heading
    line = gdf_file.readline()
    values = line.split(',')
    values = [x.strip() for x in values]

    node_attr=[]
    node_attr_type=[]
    if len(values)>2:
        for i in range(2,len(values)):
            node_attr+=[values[i].split()[0].strip()]
            node_attr_type+=[values[i].split()[1].strip()]

    # length1 = number of node attributes
    length1 = len(node_attr)

    # Read node section (until WHILE condition, which is the heading in edge section)
    line = gdf_file.readline()
    values = line.split(',')
    for i in range(len(values)):
        values[i]=values[i].strip()

    while values[0]!= "edgedef>node1 VARCHAR":

        for i in range(len(values)):
            values[i]=values[i].strip()

        G.add_node(values[0])

        attr_dict={}
        for i in range(length1):
            if node_attr_type[i]== 'VARCHAR':
                try:
                    val=int(values[i+2])
                except ValueError:
                    val=str(values[i+2])
            elif node_attr_type[i]== 'DOUBLE':
                val=float(values[i+2])
            elif node_attr_type[i]== 'INTEGER':
                val=values[i+2]
            attr_dict[node_attr[i]]=val

        for key,value in attr_dict.iteritems():
            G.node[values[0]][key]=value

        line = gdf_file.readline()
        values = line.split(',')
        for i in range(len(values)):
            values[i]=values[i].strip()

    # Deduce link attributes (and types) from heading

    edge_attr=[]
    edge_attr_type=[]

    if len(values) > 2:
        for i in range(2,len(values)):
            edge_attr+=[values[i].split(' ')[0].strip()]
            edge_attr_type+=[values[i].split(' ')[1].strip()]

    # length2 = number of node attributes
    length2 = len(edge_attr)

    line = gdf_file.readline()
    values = line.split(',')
    for i in range(len(values)):
        values[i]=values[i].strip()

    while line != '':
        G.add_edge(values[0],values[1])

        attr_dict={}

        for i in range(length2):
            # check this try-except
            if edge_attr_type[i]== 'VARCHAR':
                try:
                    val=int(values[i+2])
                except ValueError:
                    val=str(values[i+2])
            elif edge_attr_type[i]== 'DOUBLE':
                val=float(values[i+2])
            elif edge_attr_type[i]== 'INTEGER':
                val=int(values[i+2])


            attr_dict[edge_attr[i]]=val

        for key,value in attr_dict.iteritems():
            G.edge[values[0]][values[1]][key]=value

        line = gdf_file.readline()
        values = line.split(',')

        for i in range(len(values)):
            values[i]=values[i].strip()

    return G

def summary_and_graph(hashtag):

    import networkx as nx

    # Look for a GDF file
    # If there is not GDF or you do not want to use it, try with the edgelist:

    G = gdf_to_nxdigraph(hashtag)

    # If it has been possible to create a nx digraph...
    if G!=None:
        G = PageRank(G)
        # Make cartography
        plotCartography = " " # to skip plotting
        G = cartography(G,hashtag,plot=plotCartography)

    return G

def nxdigraph_to_gdf(G, path, node_attr=None, edge_attr=None, giant=False):
    import networkx as nx
    import numpy as np
    print 'to_GDF'
    # Keep Giant component if giant == True
    if giant == True:
        G = sorted(nx.connected_component_subgraphs(G), key=len, reverse=True)[0]

    # get the info for the gdf node section header:
    # - define the type of node attributes and put them in a list
    node_attr_gdf = []
    if node_attr != None:
        for i in range(len(node_attr)):
            attr = G.node[G.nodes()[0]][node_attr[i]]
            if type(attr) != int and type(attr) != float and type(attr) != np.float64:
                node_attr_gdf += ['VARCHAR']
                print 'node attr', i, ': varchar'

            elif type(attr) == int:
                node_attr_gdf += ['INTEGER']
                print 'node attr', i, ': int'

            elif type(attr) == float or type(attr) == np.float64:
                node_attr_gdf += ['DOUBLE']
                print 'node attr', i, ': double'


            else:
                # Maybe there are more options to consider apart from the three above
                print 'ERROR: node attr type not recognized'
                print attr, type(attr)

    # get the info for the gdf edge section header:
    # - define the type of edge attributes and put them in a list
    edge_attr_gdf = []
    if edge_attr != None:
        for i in range(len(edge_attr)):
            edge = G.edges()[0]
            attr = G[edge[0]][edge[1]][edge_attr[i]]
            if type(attr) != int and type(attr) != float and type(attr) != np.float64:
                edge_attr_gdf += ['VARCHAR']
                print 'edge attr', i, ': varchar'

            elif type(attr) == int:
                edge_attr_gdf += ['INTEGER']
                print 'edge attr', i, ': int'

            elif type(attr) == float or type(attr) == np.float64:
                edge_attr_gdf += ['DOUBLE']
                print 'edge attr', i, ': double'

            else:
                print 'ERROR: edge attr type not recognized'
                print attr, type(attr)

    print 'writing...'

    path='../gdf_gephi/'+path

    with open(path, 'wb') as f:

        # Write NODES section
        # Heading
        f.write('nodedef>name VARCHAR,label VARCHAR')
        if node_attr != None:
            for i in range(len(node_attr)):
                f.write(',' + node_attr[i] + ' ' + node_attr_gdf[i])

        f.write('\n')
        print 'nodes header done'

        # Elements
        for item in G.nodes():

            if node_attr != None:

                f.write(str(item) + ', ' + str(item))
                for i in range(len(node_attr)):
                    if i < len(node_attr) - 1:
                        f.write(', ' + str(G.node[item][node_attr[i]]))
                    else:
                        f.write(', ' + str(G.node[item][node_attr[i]]) + '\n')
            else:
                f.write(str(item) + ', ' + str(item) + '\n')
        print 'nodes done'

        # Write LINK section
        # Heading
        f.write('edgedef>node1 VARCHAR,node2 VARCHAR,directed BOOLEAN')
        if edge_attr != None:
            f.write(',')
            for i in range(len(edge_attr)):
                f.write(edge_attr[i] + ' ' + edge_attr_gdf[i])
                if i < len(edge_attr) - 1:
                    f.write(',')
                else:
                    f.write('\n')
        else:
            f.write('\n')
        print 'edges header done'

        # Elements
        for item in G.edges():
            if edge_attr != None:
                f.write(str(item[0]) + ' , ' + str(item[1]) + ', True, ')
                for i in range(len(edge_attr)):
                    if i < len(edge_attr) - 1:
                        f.write(str(G.edge[item[0]][item[1]][edge_attr[i]]) + ',')
                    else:
                        f.write(str(G.edge[item[0]][item[1]][edge_attr[i]]) + '\n')

            else:
                f.write(str(item[0]) + ' , ' + str(item[1]) + ', True \n')

        print 'edges done'

    return

def PageRank(G):

    import networkx as nx
    from random import randint
    # Centrality and centralization
    # PageRank centrality

    N = nx.number_of_nodes(G)
    node = randint(0,N-1)
    PRfound = 'PageRank' in G.nodes(data=True)[node][1].keys()

    if (PRfound == False):
        aux = nx.pagerank(G)
        pr = []
        for key, value in aux.iteritems():
            temp = [key, value]
            pr.append(temp)
        pr.sort(key=lambda tup: tup[1], reverse=True)
        # Set&print node attributes
        PRdic={}
        for i in range(len(pr)):
            n=pr[i][0]
            PRdic[n]=pr[i][1]
        nx.set_node_attributes(G, 'PageRank', PRdic)
        print 'PageRank done'

    return G

###############################################################################

# Write the path to the gdfs folder

from random import randint
import networkx as nx

HASHTAGS = os.listdir('../gdf_gephi/')

NoFiedler = []
NoEC = []

for topic in HASHTAGS:
    
    print topic

    # 1) Update PageRank and Cartography:
    topic = topic[:-4]
    G = summary_and_graph(topic)
    node_attr=G.nodes(data=True)[0][1].keys()

    gdf_file='../gdf_gephi/'+topic+'.gdf'
    nxdigraph_to_gdf(G,gdf_file,node_attr)
    
    # 2) Check node attributes
    N = nx.number_of_nodes(G)
    node = randint(0,N-1)
    ECfound = 'EC' in G.nodes(data=True)[node][1].keys()
    Fiedlerfound = 'Fiedler' in G.nodes(data=True)[node][1].keys()
    
    if (not ECfound):
        NoEC.append(topic)
    if (not Fiedlerfound):
        NoEC.append(topic)
        
print '   >>> Topics with no Eigenvector Centrality: '
print "\n".join(NoEC)
print '   >>> Topics with no Fiedler value: '
print "\n".join(NoFiedler)
