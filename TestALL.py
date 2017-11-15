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
        # give up

    # If it has been possible to create a nx digraph...
    if G!=None:
#        N=len(G)
#        L=len(G.edges())
#        print 'This network has',N,'nodes and',L,'links.'
#
#        # Look for the summary file
#        try:
#            filename='../summary/'+hashtag+'_summary.csv'
#            summary_file = open(filename, "r")
#            summary_file.close()
#        except:
#            print 'No summary file found.'
#            pass
#
#        # Create a summary file if required
#        #number of interacting accounts (that have/have been mentioned or RT)
#        n_interacting = len(G)
#        check_txt=0
#        #activity
#        filename='../raw_data/'+hashtag+'.txt'
#        txtfile = open(filename,'r')
#        print 'Reading file txt...'
#        check_txt=1
#
#        n_TW=0
#        n_RT=0
#        act={}
#        emitting={}
#        lc=0
#        while 1:
#            line = txtfile.readline()
#            lc+=1
#            if line=='': break
#            # each line is a TW
#            n_TW+=1
#            # n is an active user, emitting or RT
#            n = line.split()[0]
#            act[n] = 1
#            # is a RT?
#            if 'RT' in line:
#                n_RT+=1
#                if n not in emitting.keys():
#                    emitting[n] = 0
#            else:
#                emitting[n]=1
#
#
#            # is n an interacting user? (if the TW is a RT, n is always an interactin user)
#            # if not, add the isolated node
#            if n not in G.nodes():
#                G.add_node(n)
#
#            if lc%1000==0: print 'line ',lc
#        txtfile.close()
#
#        print('Read txt done')
#        for n in G.nodes():
#            if n not in act.keys():
#                act[n] = 0
#                emitting[n] = 0
#
#        # number of accounts...
#        # 1) total
#        tot_accounts = len(G)
#        # 2) active: emitting or RT
#        n_active = sum(a for a in act.values())
#        # 3) emitting but not gaining attention
#        n_loosers = tot_accounts - n_interacting
#        # 4) passive accounts that receive attention but do nothing
#        n_passive = tot_accounts - n_active
#        # 5) emitting accounts (with emitting TW)
#        n_emitting = sum(x for x in emitting.values())
#        # Set&print node attributes
#        nx.set_node_attributes(G, 'act', act)
#        nx.set_node_attributes(G, 'emitting', emitting)

        # With or without TXT file, if the summary has to be (re)written, calculate centralities
        G, IPR = centralities(G)
        # Summary File
#        filename='../summary/'+hashtag+'_summary.csv'
#        sumfile=open(filename,'wb')
#
#        if check_txt==1:
#            print >>sumfile,"tot,active,interacting,passive,emitting,loosers,tot TW, RT,n_ORIGINAL, IPR"
#            print >>sumfile,tot_accounts,",",n_active,",",n_interacting,",",n_passive,",",n_emitting,",",n_loosers,",",n_TW,",",n_RT,",",n_TW-n_RT,",",IPR
#            sumfile.close()
#        else:
#            print >>sumfile,"Interacting, IPR"
#            print >>sumfile,len(G),",",IPR
#            sumfile.close()


        # Make cartography
        plotCartography = " "
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

def centralities(G):

    import networkx as nx
    # Centrality and centralization
    # PageRank centrality
    if len(G)>5000:
        check=True
        if len(G)>50000:
            checkPR=True
        else: checkPR=False
    else:
        check=False
        checkPR=False

    go_on='Y'
    while(go_on=='Y'):
        if checkPR==True:
            go_on=raw_input("Do yo want to calculate the PageRank centrality? Y/N\n")[0]
            if go_on=='y': go_on='Y'

        if go_on!='Y': break
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
        go_on='N'

    # Eigenvector Centralization (IPR)
    go_on = 'Y'
    while (go_on=='Y'):
        if check == True:
            go_on='Y'

        Gcc = sorted(nx.weakly_connected_component_subgraphs(G), key=len, reverse=True)
        G0 = Gcc[0] #largest connected subgraph
        EC = nx.eigenvector_centrality(G0)
        IPR = 1. / sum([v ** 4 for v in EC.values()]) / float(len(G0))
        for n in G.nodes():
            if n not in EC.keys():
                EC[n]=0
        # Set&print node attributes
        nx.set_node_attributes(G, 'EC', EC)
        print 'Eigenvector Centrality done'
        print 'IPR=',round(IPR,4)
        go_on = 'N'

    # Fiedler eigenvector
    go_on = 'Y'
    while (go_on == 'Y'):
        if check == True:
            go_on = 'Y'
        if go_on != 'Y': break
        from networkx.linalg.algebraicconnectivity import fiedler_vector
        import numpy as np
        from numpy import linalg as la
        Gud=G0.to_undirected()
        #FV=fiedler_vector(Gud)
        A = nx.adjacency_matrix(Gud, nodelist=Gud.nodes())
#        D = np.diag(np.ravel(np.sum(A, axis=1)))
        D = np.diag(np.ravel(A.sum()))
        L = D - A
        l, U = la.eigh(L)
        FV = U[:, 1]

        # Set&print node attributes
        FVdic={}
        for i in range(len(FV)):
            n=Gud.nodes()[i]
            FVdic[n]=FV.item((i,0))
        for n in G.nodes():
            if n not in FVdic.keys():
                FVdic[n]=0
        nx.set_node_attributes(G,'Fiedler',FVdic)
        print 'Fiedler eigenvector done'
        go_on = 'N'

    return G,IPR


###############################################################################

# Write the path to the gdfs folder
    
HASHTAGS = os.listdir('../gdf_gephi/')

for topic in HASHTAGS:
    
    print topic
    
    topic = topic[:-4]
    G = summary_and_graph(topic)
    print G
    node_attr=G.nodes(data=True)[0][1].keys()
    
    gdf_file='../gdf_gephi/'+topic+'.gdf'
    nxdigraph_to_gdf(G,gdf_file,node_attr)