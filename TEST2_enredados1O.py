from TEST2_enredados1O import *

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
            go_on = raw_input("Do yo want to calculate the eigenvector centrality? Y/N\n")[0]
            if go_on=='y': go_on='Y'
        if go_on != 'Y':
            IPR='Not calculated'
            break

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
            go_on = raw_input("Do yo want to calculate the Fiedler eigenvector? Y/N\n")[0]
            if go_on == 'y': go_on = 'Y'
        if go_on != 'Y': break
        from networkx.linalg.algebraicconnectivity import fiedler_vector
        import numpy as np
        from numpy import linalg as la
        Gud=G0.to_undirected()
        #FV=fiedler_vector(Gud)
        A = nx.adjacency_matrix(Gud, nodelist=Gud.nodes())
        D = np.diag(np.ravel(np.sum(A, axis=1)))
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

def summary_and_graph(hashtag):

    import networkx as nx

    # Look for a GDF file
    check_gdf=0
    try:
        # Open GDF file
        G=gdf_to_nxdigraph(hashtag)
        check_gdf =int(raw_input("GDF file found. Enter 1 if you want to use it, 0 if you prefer to overwrite it."))
    except:
        print 'GDF file not found.'
        pass

    # If there is not GDF or you do not want to use it, try with the edgelist:
    if check_gdf==0:
        print 'Trying to create graph from edgelist...'
        try:
            # Open EDGELIST file
            filename='../raw_data/'+hashtag+'.edges'
            list_file = open(filename,"r")
            G = nx.read_edgelist(list_file, delimiter='\t', create_using=nx.DiGraph(), data=False)
            list_file.close()
            print 'Read edgelist done. Basic graph created.'

        except:
            print 'No input file found.'
            G=None
            # give up

    # If it has been possible to create a nx digraph...
    if G!=None:
        N=len(G)
        L=len(G.edges())
        print 'This network has',N,'nodes and',L,'links.'

        # Look for the summary file
        check_summary=0
        try:
            filename='../summary/'+hashtag+'_summary.csv'
            summary_file = open(filename, "r")
            summary_file.close()
            check_summary=int(raw_input('There is already a summary file. Enter 1 if you want to keep it, 0 if you want to overwrite it.'))
        except:
            print 'No summary file found.'
            pass

        # Create a summary file if required
        if check_summary==0:
            #number of interacting accounts (that have/have been mentioned or RT)
            n_interacting = len(G)
            check_txt=0
            try:
                #activity
                filename='../raw_data/'+hashtag+'.txt'
                txtfile = open(filename,'r')
                print 'Reading file txt...'
                check_txt=1

                n_TW=0
                n_RT=0
                act={}
                emitting={}
                lc=0
                while 1:
                    line = txtfile.readline()
                    lc+=1
                    if line=='': break
                    # each line is a TW
                    n_TW+=1
                    # n is an active user, emitting or RT
                    n = line.split()[0]
                    act[n] = 1
                    # is a RT?
                    if 'RT' in line:
                        n_RT+=1
                        if n not in emitting.keys():
                            emitting[n] = 0
                    else:
                        emitting[n]=1


                    # is n an interacting user? (if the TW is a RT, n is always an interactin user)
                    # if not, add the isolated node
                    if n not in G.nodes():
                        G.add_node(n)

                    if lc%1000==0: print 'line ',lc
                txtfile.close()

                print('Read txt done')
                for n in G.nodes():
                    if n not in act.keys():
                        act[n] = 0
                        emitting[n] = 0

                # number of accounts...
                # 1) total
                tot_accounts = len(G)
                # 2) active: emitting or RT
                n_active = sum(a for a in act.values())
                # 3) emitting but not gaining attention
                n_loosers = tot_accounts - n_interacting
                # 4) passive accounts that receive attention but do nothing
                n_passive = tot_accounts - n_active
                # 5) emitting accounts (with emitting TW)
                n_emitting = sum(x for x in emitting.values())
                # Set&print node attributes
                nx.set_node_attributes(G, 'act', act)
                nx.set_node_attributes(G, 'emitting', emitting)

            except:
                print 'TXT file not found.'
                pass
            # With or without TXT file, if the summary has to be (re)written, calculate centralities
            G, IPR = centralities(G)
            # Summary File
            filename='../summary/'+hashtag+'_summary.csv'
            sumfile=open(filename,'wb')

            if check_txt==1:
                print >>sumfile,"tot,active,interacting,passive,emitting,loosers,tot TW, RT,n_ORIGINAL, IPR"
                print >>sumfile,tot_accounts,",",n_active,",",n_interacting,",",n_passive,",",n_emitting,",",n_loosers,",",n_TW,",",n_RT,",",n_TW-n_RT,",",IPR
                sumfile.close()
            else:
                print >>sumfile,"Interacting, IPR"
                print >>sumfile,len(G),",",IPR
                sumfile.close()


        # Make cartography
        check_cart='y'
        if len(G)>=50000:
            check_cart=raw_input("Do you want to make a cartography? ('Y' or 'N') ")[0]
        if check_cart=='Y' or check_cart=='y':
            plotCartography = raw_input("  - @cartography: Which degree kind are you plotting? ('In','Out' or 'All'): ")
            print " "
            G = cartography(G,hashtag,plot=plotCartography)

    return G,check_gdf
#G,c=summary_and_graph('pippo')

def do_it(hashtag=None):
    if hashtag==None:
        hashtag = raw_input("Enter a hashatag (w/o #): ")
    G,check_gdf=summary_and_graph(hashtag)
    if G!=None:
        node_attr=G.nodes(data=True)[0][1].keys()
        if check_gdf==1: hashtag=hashtag+str('_bis')
        gdf_file='../gdf_gephi/'+hashtag+'.gdf'
        nxdigraph_to_gdf(G,gdf_file,node_attr)
    return
#do_it('500retenidos')

def comparison(hashtag1=None,hashtag2=None,ntop=50):

    if hashtag1==None:
        hashtag1 = raw_input("Enter the 1st hashatag (w/o #): ")
    if hashtag2 == None:
        hashtag2 = raw_input("Enter the 2nd hashatag (w/o #): ")

    G1=gdf_to_nxdigraph(hashtag1)
    G2=gdf_to_nxdigraph(hashtag2)

    N1=set(G1.nodes())
    N2=set(G2.nodes())
    I=N1.intersection(N2)
    n1=len(N1)
    n2=len(N2)
    f1=float(len(I))/float((len(N1)))
    f2=float(len(I))/float((len(N2)))
    J=float(len(I))/(n1+n2-len(I))
    print 'hashtag1:',hashtag1, 'hashtag2=',hashtag2,'\n'
    print 'N. user hashtag1:',n1, 'fraction shared with hashtag2:',round(f1,4)
    print 'N. user hashtag2:',n2, 'fraction shared with hashtag2:', round(f2,4)
    print 'Jaccard Index', round(J,4),'\n'

    nodes1=[]
    for n in G1.nodes(data=True): nodes1+=[(n[0], n[1]['PageRank'])]
    nodes1=sorted(nodes1,key=lambda x: x[1],reverse=True)

    nodes2=[]
    for n in G2.nodes(data=True): nodes2+=[(n[0], n[1]['PageRank'])]
    nodes2=sorted(nodes2,key=lambda x: x[1],reverse=True)
    Itop=[]
    ntop=int(ntop)
    while len(Itop)==0:
        top1=nodes1[:ntop]
        dtop1=dict(top1)
        top2=nodes2[:ntop]
        dtop2=dict(top2)

        Itop=set(dtop1.keys()).intersection(set(dtop2.keys()))
        ntop+=1

    print 'hashtag1:',hashtag1
    for i in range(len(nodes1)):
        if nodes1[i][0] in Itop:
            print i+1, nodes1[i][0],
            for attr in G1.node[nodes1[i][0]].keys():
                print attr,'=',G1.node[nodes1[i][0]][attr],
            print
            #round(nodes1[i][1],4),'ACT=',G1.node[nodes1[i][0]]['act']
    print
    print 'hashtag2:',hashtag2
    for i in range(len(nodes2)):
        if nodes2[i][0] in Itop:
            print i+1, nodes2[i][0],
            #round(nodes2[i][1],4),'ACT=',G2.node[nodes2[i][0]]['act']
            for attr in G2.node[nodes2[i][0]].keys():
                print attr,'=',G2.node[nodes2[i][0]][attr],
            print

    print "===============\n"
    return

def users_ranking(hashtag=None,measure='PageRank',ntop=20):

    if hashtag==None:
        hashtag = raw_input("Enter the hashatag (w/o #): ")

    G=gdf_to_nxdigraph(hashtag)

    nodes=[]
    for n in G.nodes(data=True): nodes+=[(n[0], n[1][measure])]
    nodes=sorted(nodes,key=lambda x: x[1],reverse=True)

    print 'hashtag:',hashtag
    tab=[]
    for i in range(ntop):
        if G.node[nodes[i][0]]['act']==1:
            a='SI'
        else: a = 'NO'
        tab+=[str(i+1)+' '+str(nodes[i][0])+' ACT='+G.node[nodes[i][0]]['act']+'    '+str(round(nodes[i][1],4))]
    for i in range(ntop):
        print tab[i]
    return

def one_user(hlist=None,user=None,measure='PageRank'):
    import os
    from os import listdir
    from os.path import isfile, join,splitext

    if user==None:
        user = raw_input("Enter the username: ")

    if hlist==None:
        nh = raw_input("How many hashtags do you want to check?")
        hlist = []
        if nh!='all':
            nh=int(nh)
            for i in range(nh):
                hlist += [raw_input("Enter a hashatag (w/o #): ")]
        else:
            print "Let's check all the hashtags."
            hlist = [os.path.splitext(f)[0] for f in listdir('../gdf_gephi/') if isfile(join('../gdf_gephi/', f)) and f.endswith(".gdf")]

    prop = {}

    for hashtag in hlist:
        G=gdf_to_nxdigraph(hashtag)

        node_attr = G.nodes(data=True)[0][1].keys()
        nodes = []

        if measure in node_attr:
            for n in G.nodes(data=True): nodes+=[(n[0], n[1][measure])]
        else:
            print measure,'is not a node attribute in',hashtag,'. Ordering nodes by in degree.'
            for n in G.nodes(data=True): nodes+=[(n[0], G.in_degree(n[0]))]


        nodes=sorted(nodes,key=lambda x: x[1],reverse=True)

        print 'hashtag:',hashtag
        check=0
        n=0
        k=-1
        while n!=user:
            k+=1
            if k==len(G):
                print 'User not found'
                check=1
                break
            n=nodes[k][0]
        if check==0:
            print nodes[k][0],':''rank=', k+1,'k_in=',G.in_degree(user),'k_out=',G.out_degree(user),
            if 'rank' not in prop.keys():
                prop['rank']={}
            prop['rank'][hashtag]=k+1
            if 'in_degree' not in prop.keys():
                prop['in_degree']={}
            prop['in_degree'][hashtag]=G.in_degree(user)
            if 'out_degree' not in prop.keys():
                prop['out_degree']={}
            prop['out_degree'][hashtag]=G.out_degree(user)

            for attr in node_attr:
                print attr,'=',G.node[user][attr],
                if attr not in prop.keys():
                    prop[attr]={}
                prop[attr][hashtag]=G.node[user][attr]
            print

    if nh=='all':
        f = open('../users/'+user + '_rank.csv', 'w')
        print 'Writing csv...'
        for attr in prop.keys():
            print >>f,attr
            print attr
            for h in prop[attr].keys():
                print >>f, '"',h,'"',prop[attr][h]
                print h, prop[attr][h]
            print >>f
        f.close()
    print 'done.'
    return

def join(hashtag_list):
    import networkx as nx
    G=nx.DiGraph()
    tag = {}
    for h in hashtag_list:
        F = gdf_to_nxdigraph(h)
        tag[h]={}
        for n in F.nodes():
            tag[h][n] = int(F.node[n]['act'])+int(F.node[n]['emitting'])
        #G=nx.compose(G,F)
        G.add_nodes_from(F.nodes(data=False))
        G.add_edges_from(F.edges(data=True))
    for h in hashtag_list:
        for n in G.nodes():
            if n not in tag[h].keys(): tag[h][n]=-1
        nx.set_node_attributes(G, str(h), tag[h])

    print 'Merged!'

    print 'Tot accounts=',len(G),'Tot Interactions=',len(G.edges())
    n_act=0
    n_emit=0
    for n in G.nodes():
        for h in hashtag_list:
            if G.node[n][h]>0:
                n_act+=1
            if G.node[n][h]>1:
                n_emit+=1
    print 'N. active accounts=',n_act
    print 'N. emitting accounts=',n_emit

    if len(G)>5000:check=True
    else: check=False

    G=centralities(G,check)

    return G


def cartography(G_0, hashtag, plot = None):

    # Script that takes a graph (G) and updates a gdf file with the within
    # module degree (Z), and he participation coefficient (P). Based on these
    # the node role in the network is also computed and added to the gdf.
    # The returning object is the updated .gdf file and a cartography plot

    import community
    import numpy as np
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches

    def links_in_module(G, node, module, In=True, Out=True):

        if (In == True) and (Out == True):
            Neighbors = set(G.predecessors(node) + G.successors(node))
            numlinks = sum(1 for neighbor in Neighbors if neighbor in module)
        elif (In == True) and (Out == False):
            numlinks = sum(1 for neighbor in G.predecessors(node) if neighbor in module)
        elif (In == False) and (Out == True):
            numlinks = sum(1 for neighbor in G.successors(node) if neighbor in module)

        return numlinks

    def FindRole(G,n_i,degree="All"):

        # Function that classify the nodes takin into account its z and P values
        role = None
        if (degree=="All"):

            if G.node[n_i]['z'] < 2.5:
                if (G.node[n_i]['P_coef'] < 0.05):
                    role = "R1"
                elif (G.node[n_i]['P_coef'] >= 0.05) and (G.node[n_i]['P_coef'] < 0.625):
                    role = "R2"
                elif (G.node[n_i]['P_coef'] >= 0.625) and (G.node[n_i]['P_coef'] < 0.8):
                    role = "R3"
                elif (G.node[n_i]['P_coef'] >= 0.8):
                    role = "R4"
            elif G.node[n_i]['z'] >= 2.5:
                if (G.node[n_i]['P_coef'] < 0.3):
                    role = "R5"
                elif (G.node[n_i]['P_coef'] >= 0.3) and (G.node[n_i]['P_coef'] < 0.75):
                    role = "R6"
                elif (G.node[n_i]['P_coef'] >= 0.75):
                    role = "R7"
            G.node[n_i].update({'role' : role})

        elif (degree=="In"):

            if G.node[n_i]['z_In'] < 2.5:
                if (G.node[n_i]['P_coef'] < 0.05):
                    role = "R1"
                elif (G.node[n_i]['P_coef'] >= 0.05) and (G.node[n_i]['P_coef'] < 0.625):
                    role = "R2"
                elif (G.node[n_i]['P_coef'] >= 0.625) and (G.node[n_i]['P_coef'] < 0.8):
                    role = "R3"
                elif (G.node[n_i]['P_coef'] >= 0.8):
                    role = "R4"
            elif G.node[n_i]['z_In'] >= 2.5:
                if (G.node[n_i]['P_coef'] < 0.3):
                    role = "R5"
                elif (G.node[n_i]['P_coef'] >= 0.3) and (G.node[n_i]['P_coef'] < 0.75):
                    role = "R6"
                elif (G.node[n_i]['P_coef'] >= 0.75):
                    role = "R7"
            G.node[n_i].update({'In_role' : role})

        elif (degree=="Out"):

            if G.node[n_i]['z_Out'] < 2.5:
                if (G.node[n_i]['P_coef'] < 0.05):
                    role = "R1"
                elif (G.node[n_i]['P_coef'] >= 0.05) and (G.node[n_i]['P_coef'] < 0.625):
                    role = "R2"
                elif (G.node[n_i]['P_coef'] >= 0.625) and (G.node[n_i]['P_coef'] < 0.8):
                    role = "R3"
                elif (G.node[n_i]['P_coef'] >= 0.8):
                    role = "R4"
            elif G.node[n_i]['z_Out'] >= 2.5:
                if (G.node[n_i]['P_coef'] < 0.3):
                    role = "R5"
                elif (G.node[n_i]['P_coef'] >= 0.3) and (G.node[n_i]['P_coef'] < 0.75):
                    role = "R6"
                elif (G.node[n_i]['P_coef'] >= 0.75):
                    role = "R7"

            G.node[n_i].update({'Out_role' : role})

    def InCartoPlot(G1,node,Z):

        Z_range = (-100,100)
        inplot = False
        if (G1.node[node][Z] > Z_range[0])  and (G1.node[node][Z] < Z_range[1]):
            inplot = True

        return inplot
    
    def Relevance(G,node, K = 'All'):
        
        Z = None
        if (K == 'All'): Z = 'z'
        elif (K == 'In'): Z = 'z_In'
        elif (K == 'Out'): Z = 'z_Out'
        
        z_node = G.node[node][Z]
        P_node = G.node[node]["P_coef"]
        relevance = np.log(G.degree(node))*z_node*P_node # Relevance is defined to highlight outliars from each module
        
        return relevance

    def FindKeyRoles(G1,module1, K1 = 'All',Log = True):
        import operator

        Z = None
        if (K1 == 'All'): Z = 'z'
        elif (K1 == 'In'): Z = 'z_In'
        elif (K1 == 'Out'): Z = 'z_Out'

        R = {}
        Sizes = []
        for node in module1:
            if InCartoPlot(G1,node,Z):
                relevance = Relevance(G1, node, K = K1)
                R.update({node:relevance})
            Sizes.append(G1.degree(node))

        ScaleSqrt = 600
        ScaleLog = 15000
        NormSizes = [float(i)/sum(Sizes) for i in Sizes]
        LogNormSizes = [ScaleLog*np.log(x+1) for x in NormSizes]
        SqrtNormSizes = [ScaleSqrt*2*np.sqrt(x)/(x+1) for x in NormSizes]

        Key1 = max(R.iteritems(), key=operator.itemgetter(1))[0]
        del R[Key1]
        Key2 = max(R.iteritems(), key=operator.itemgetter(1))[0]
        del R[Key2]
        Key3 = max(R.iteritems(), key=operator.itemgetter(1))[0]
        del R[Key3]

        HUBS = [[Key1, Key2, Key3],
                 [G1.node[Key1]['P_coef'],G1.node[Key2]['P_coef'],G1.node[Key3]['P_coef']],
                 [G1.node[Key1][Z],G1.node[Key2][Z],G1.node[Key3][Z]]]

        if(Log == True):
            N = LogNormSizes
        else:
            N = SqrtNormSizes

        return HUBS, N

    def checkNAN(LIST):

        import math

        NANfound = False
        for item in LIST:
            NANfound = math.isnan(item)
            if NANfound:
                print "NAN found"
                break

        print "NAN has been found: ", NANfound
        return

    ###########################################################################
    #                                  SETTING UP                             #
    ###########################################################################

    # Removing isolated nodes
    for n in G_0.nodes():
        if G_0.degree(n) == 0:
            G_0.remove_node(n)
    G = G_0

    ###########################################################################
    #                         FINDING COMMUNITIES                             #
    ###########################################################################

    # Computing the best partition:
    #   - The partition is defined to be computed for undirected graphs.
    G_undirected = G.to_undirected()
    partition = community.best_partition(G_undirected)
    numPartitions = max(partition.values()) + 1
    MODULES = [[] for _ in range(numPartitions)]

    # MODULES is a list continaing sublists representing the different modules
    # in the network. Each sublist contains the labels of all nodes of that module

    for index, value in enumerate(partition.values()):
        MODULES[value].append(partition.keys()[index])

    MODULES.sort(reverse=True,key=len)

    ###########################################################################
    #                            UPDATING GDF                                 #
    ###########################################################################

    # Checking for the attributes of the first node in G to see if the graph
    # has already the cartography attributes
    FirstUser = G.nodes(data=True)[0]
    if ('community_id' in FirstUser[1]):
        print ("       >> Node attributes in place, proceeding with the plot.")
        print (" ")
    else:
        PlotCeiling = 0
        PlotFloor = 0

        for module in MODULES:

            ALL_Ks = [links_in_module(G,node_i,module, In=True, Out=True) for node_i in module]
            ALL_Ks_In = [links_in_module(G,node_i,module, In=True, Out=False) for node_i in module]
            ALL_Ks_Out = [links_in_module(G,node_i,module, In=False, Out=True) for node_i in module]

            avg_Ks = np.mean(ALL_Ks)
            avg_Ks_In = np.mean(ALL_Ks_In)
            avg_Ks_Out = np.mean(ALL_Ks_Out)

            std_Ks = np.std(ALL_Ks)
            std_Ks_In = np.std(ALL_Ks_In)
            std_Ks_Out = np.std(ALL_Ks_Out)


            for node in module:

                G.node[node].update({'community_id' : partition[node]})

                # i) Computing within module degree (Z). Given that we are working with a DiGraph
                #    this parameter depends on the kind of links you are taking into account: nodes
                #    that go in the node, these going out of the node, or both of them.

                # Number of links of node to other nodes in its module
                k_s_i = links_in_module(G,node,module, In=True, Out=True)
                k_s_i_In = links_in_module(G,node,module, In=True, Out=False)
                k_s_i_Out = links_in_module(G,node,module, In=False, Out=True)

                dummy_z = -3 # Dummy value to avoid NaN problem

                if (std_Ks != 0.0): zi = (k_s_i - avg_Ks)/std_Ks
                else: zi = dummy_z

                if (std_Ks_In != 0.0): zi_In = (k_s_i_In - avg_Ks_In)/std_Ks_In
                else: zi_In = dummy_z

                if (std_Ks_Out != 0.0): zi_Out = (k_s_i_Out - avg_Ks_Out)/std_Ks_Out
                else: zi_Out = dummy_z

                G.node[node].update({'z' : zi})
                G.node[node].update({'z_In' : zi_In})
                G.node[node].update({'z_Out' : zi_Out})

                maxZ = int(max(zi,zi_In,zi_Out))
                minZ = int(min(zi,zi_In,zi_Out))
                if  (PlotCeiling < maxZ): PlotCeiling = maxZ
                if  (PlotFloor > minZ): PlotFloor = minZ

                # ii) Computing participation coefficient (P)

                ki = G.degree(node)
                x = sum((float(links_in_module(G,node,mod))/ki)**2 for mod in MODULES)
                Pi = 1 - x

                G.node[node].update({'P_coef' : Pi})

                # iii) Finding node role based on Z and P

                FindRole(G,node)
                FindRole(G,node, degree = "In")
                FindRole(G,node, degree = "Out")

    ###########################################################################
    #                            PLOTTING TIME                                #
    ###########################################################################

    DegreeKinds = ['In','Out','All']
    if (plot in DegreeKinds):

        zKind = None
        if (plot == 'In'): zKind = 'z_In'
        if (plot == 'Out'): zKind = 'z_Out'
        if (plot == 'All'): zKind = 'z'

        PlotCeiling = int(max([n[1][zKind] for n in G.nodes(data=True)]))
        PlotFloor = int(min([n[1][zKind] for n in G.nodes(data=True)]))

        # Drawing cartography plot background
        fig, ax1 = plt.subplots(1,1, figsize = (20,12))
        ax1 = DrawCarto(ax1,PlotCeiling,PlotFloor,hashtag)
        # ax2 is only to add the legend
        ax2 = fig.add_axes([0,0,1,1],frameon=False)
        ax2.patch.set_alpha(0.0)
        ax2.set_yticks([])
        ax2.set_xticks([])
        # For readability, only plotting 6 biggest categories
        Categories2plot = min(numPartitions,6)
        # Colors for the categories
        C = ["Tomato","DarkTurquoise","DeepPink","SlateBlue","SpringGreen","Teal"]
        c_index = 0
        # Vectors for the legend of relevant users
        CategoryLeaders = []
        Numbers = [n+1 for n in range(3*Categories2plot)]
        
        OTHERS = {}
        RELEVANCES = []

        for module in MODULES:
            # MODULES is ordered decreasingly, so the first modules are printed
            # with color for highlighting
            
            Ps = [G.node[n]['P_coef'] for n in module]               
            Zs = [G.node[n][zKind] for n in module]
            
            if (c_index < Categories2plot):

                big3,ModuleSizes = FindKeyRoles(G, module, K1 = plot)
                RELEVANCES = RELEVANCES + [Relevance(G,user,K=plot) for user in big3[0]]
                
                Color = C[c_index]
                Order = 100

                # Each relevant user is labelled with a number to be identified in
                # the plot. Index of Big3 0,1 and 2 are: names, Ps and Zs respectively
                nums = Numbers[3*c_index:3*(1+c_index)]
                for num, label, x, y in zip(nums, big3[0],big3[1],big3[2]):
                    ax1.annotate(str(num),xy=(x, y), xytext=(-10, 0),
                                 textcoords='offset points',fontsize = 16,zorder=150)
                NameNums = []
                for i in range(3):
                    NameNums.append(big3[0][i] + " (" + str(nums[i]) + ")")
                CategoryLeaders.append(" \n".join(NameNums))

                c_index += 1

            # The rest of the communities are printed in light gray
            else:
                
                for user in module:
                    r = Relevance(G,user,K=plot)
                    if r > np.mean(RELEVANCES):
                        OTHERS.update({user:{'coord':(G.node[user]['P_coef'],G.node[user][zKind]), 'num': 0}})
                        
                Color = 'LightGray'
                Order = 20

            ax1.scatter(Ps,Zs, c = Color, s=ModuleSizes, lw = 0, zorder=Order)
        
        title = "#" + hashtag + " (" + plot + " Degree)"


        PATCHES = []
        for m in range(Categories2plot):
            PATCHES.append(mpatches.Patch([],[],color=C[m]))
            
        
        ax1.set_title(title, fontsize=36)
        legendRelevant = ax2.legend(handles=PATCHES,labels=CategoryLeaders,title="TOP3 relevant users by community",fontsize=18,frameon=False,
                            ncol=Categories2plot,loc="upper center",bbox_to_anchor=(0.5,-0.02))
        plt.setp(legendRelevant.get_title(),fontsize=22)
        
        # Plotting other relevant users that are not in the TOP communities
        OtherPATCHES = [mpatches.Patch([],[],color='LightGray')]
        finalN = len(Numbers) + len(OTHERS) + 1
        OtherNums = range(len(Numbers) + 1,finalN)
        OtherWNum = []
        for i in range(len(OtherNums)):
            name = OTHERS.keys()[i]
            number = str(OtherNums[i])
            OtherWNum.append(name + " (" + number + ")")
            OTHERS[name]['num'] = number
        Others = [" \n".join(OtherWNum)]
        for name in OTHERS.keys():
            x = OTHERS[name]['coord'][0]
            y = OTHERS[name]['coord'][1]
            ax1.annotate(str(OTHERS[name]['num']),xy=(x, y), xytext=(-10, 0),
                         textcoords='offset points',fontsize = 16,zorder=150)
       
        #Drawing node distribution
        axPercents = fig.add_axes([0.082, 0.64, 0.22, 0.28]) # [relX, relY, relWidth, relHeight]

        legendOthers = ax1.legend(handles=OtherPATCHES,labels=Others,title="Other Users",fontsize=18,frameon=False,
                            ncol=1,loc="upper left",bbox_to_anchor=(1.0,1.0))
        plt.setp(legendOthers.get_title(),fontsize=22)

        P = TopicBreakdown(G,hashtag,plot = plot)
        axPercents = DrawCarto(axPercents,8,-2,hashtag,Percents=P)
        axPercents.patch.set_alpha(0.2)
            
        fig.tight_layout()
        ImageName = "../summary/" + "carto_" + hashtag + "_" + plot + ".pdf"
        fig.savefig(ImageName, bbox_inches='tight',format='pdf',dpi = fig.dpi)

    return G

def DrawCarto(ax,Ceiling,Floor,hashtag, Percents = None, user = None):

    import matplotlib.patches as patches

    # Function that draws the background of the cartography (quadrants and names).
    # If hashtag is given, the summary of the cartography is given in an smaller plot

    #Drawing background regions
    Ceiling += 3 #Adding an offset for better readability
    h1 = Ceiling - 2.5
    h2 = 2.5 - Floor

    #Preparing the vectors with the roles and their position

    Roles = ['(R1)',' Peripheral \n node (R2)',' Connector \n node (R3)',' Kinless \n node (R4)',
             ' Provincial \n hub (R5)',' Connector \n hub (R6)',' Kinless \n hub (R7)']
    Roles_Short = ['R1','R2','R3','R4','R5','R6','R7']
    Q = [((0.0, Floor), 0.05, h2), ((0.05, Floor), 0.575, h2), ((0.625, Floor), 0.175, h2), ((0.8, Floor), 0.2, h2),
                 ((0.0, 2.5), 0.3, h1), ((0.3,2.5), 0.45, h1), ((0.75,2.5), 0.25, h1)]
    numQ = len(Q)

    color1 = 'DarkGray'
    fs = 24

    #   If Percents is not NULL, then this function computes the inset with the total node distribution

    if Percents != None:
        color1 = 'Gray'
    #        color2 = 'LightGray'
    #        ax.text(0.5, float(Ceiling + Floor)/2, '#' + hashtag, horizontalalignment = 'center', verticalalignment = 'center',
    #         color = color2, weight = "bold",fontsize=52, style='normal', bbox={'facecolor':'white', 'alpha':0.0, 'pad':0})
        if user == None: ax.set_title('Total node distribution', fontsize=16)
        fs = 16
        for r in range(numQ):
            Roles_Short[r] = Roles_Short[r] + '\n' + str(Percents[r])[:4] + '%'

        ax.set_yticks([])
        ax.set_xticks([])
    else:
        Xticks = [0.0,0.2,0.4,0.6,0.8,1.0]
        nYticks = (Ceiling - Floor)/5
        Yticks = range(Floor,Ceiling)[::nYticks]
        ax.set_xticklabels(Xticks,fontsize = 18)
        ax.set_yticklabels(Yticks,fontsize = 18)
        ax.set_xticks(Xticks)
        ax.set_yticks(Yticks)
        ax.set_xlabel('Participation coefficient, P',fontsize=28)
        ax.set_ylabel('Within-module degree, Z',fontsize=28)


    for r in range(numQ):
        centerQX = Q[r][0][0] + Q[r][1]/2
        centerQY = Q[r][0][1] + Q[r][2]/2
        ax.add_patch(patches.Rectangle(Q[r][0], Q[r][1], Q[r][2], facecolor='white', edgecolor="Gray", ls = '--', lw = 0.7, alpha=0.1))
        if Percents != None:
            # Filling each quadrant proportional to the percentage
            ax.add_patch(patches.Rectangle(Q[r][0], Q[r][1], Percents[r]/100*Q[r][2], facecolor="#FFE4A3", edgecolor="OrangeRed",lw = 0.0, alpha=0.7))
            ax.text(centerQX, centerQY, Roles_Short[r] , horizontalalignment = 'center', verticalalignment = 'center',
                     color = color1, weight = "bold",fontsize= fs, style='normal', bbox={'facecolor':'white', 'alpha':0.0, 'pad':0})
        else:
            ax.text(centerQX, centerQY, Roles[r] , horizontalalignment = 'center', verticalalignment = 'center',
                     color = color1, weight = "bold",fontsize= fs, style='normal', bbox={'facecolor':'white', 'alpha':0.0, 'pad':0})

    ax.set_xlim(0,1)
    ax.set_ylim(Floor,Ceiling)

    return ax

def TopicBreakdown(G,hashtag,plot='All'):
    
    import csv

    # Function that takes a topic/hashtag and returns an schema of how are
    # all the nodes of the network for that topic, distributed among quadrants.
    # This plot can be computed for any of the three kinds of degrees (In, Out, All)

    role =  None
    if plot == 'All': role = 'role'
    elif plot == 'In': role = 'In_role'
    elif plot == 'Out': role = 'Out_role'

    Rnames = ['R1','R2','R3','R4','R5','R6','R7']
    R = [0,0,0,0,0,0,0]
    for node in G.nodes():
        rol = G.node[node][role]
        index = int(rol[1:]) - 1
        R[index] += 1
    N = len(G.nodes())
    
    P = [float(r)/N*100 for r in R]
    
    P2w = [str(num)[:5] for num in P]
    filename = '../summary/' + hashtag + '_summary_carto%.csv'
    with open(filename,'wb') as f:
        writer = csv.writer(f)
        writer.writerow(Rnames)
        writer.writerow(P2w)

    return P

def UserBreakdown(username, plot='All'):

    import matplotlib.pyplot as plt
    import os
    import numpy as np

    H = os.listdir('../gdf_gephi/')
    Hashtags = [h[:-4] for h in H]
    role =  None
    zKind = None
    if plot == 'All':
        role = 'role'
        zKind = 'z'
    elif plot == 'In':
        role = 'In_role'
        zKind = 'z_In'
    elif plot == 'Out':
        role = 'Out_role'
        zKind = 'z_Out'

    Roles = [0,0,0,0,0,0,0]
    T = 0
    Ps = []
    Zs = []
    Ks = []
    for topic in Hashtags:
        # Look for a GDF file
        G = gdf_to_nxdigraph(topic)
        if username in G.nodes():
            user_role = G.node[username][role]
            index = int(user_role[1:]) - 1
            Roles[index] += 1
            T += 1
            Ps.append(float(G.node[username]['P_coef']))
            Zs.append(float(G.node[username][zKind]))
            Ks.append(G.degree(username))

    P = [float(r)*100/T for r in Roles]

    fig = plt.figure(figsize=(10,6))
    ax1 = fig.add_subplot(111)

    for label, x, y in zip(Hashtags,Ps,Zs):
        ax1.annotate('#' + label,xy=(x, y), xytext=(x, y - 0.1),
        #                     arrowprops=dict(arrowstyle="->", connectionstyle="arc3"),
                     textcoords='data',fontsize = 12,zorder=150)

    kMax = max(Ks)
    KsNorm = [float(k)/kMax for k in Ks]
    KsCale = [2000*np.sqrt(x/(x+0.1)) for x in KsNorm]
    Ceiling = int(max(Zs)) + 3
    Floor = int(min(Zs,-3))
    ax1 = DrawCarto(ax1,Ceiling,Floor,topic,Percents=P,user=username)
    Ceiling += 3 # This happens in DrawCarto
    ax1.scatter(Ps,Zs,s = KsCale, c= '#F26267', edgecolor = 'Crimson', lw = 1.0, zorder=100)
    ax1.text(0.5, (Ceiling + Floor)/2, '@' + username , zorder = 0, horizontalalignment = 'center', verticalalignment = 'center',
                     color = 'Gray', weight = "bold",fontsize= 62, style='normal', bbox={'facecolor':'white', 'alpha':0.0, 'pad':8})
    ax1.set_title('User Breakdown (' + plot + ' degree)',fontsize = 24)
    Xticks = [0.0,0.2,0.4,0.6,0.8,1.0]
    nYticks = (Ceiling - Floor)/5
    Yticks = range(Floor,Ceiling)[::nYticks]
    ax1.set_xticklabels(Xticks,fontsize = 18)
    ax1.set_yticklabels(Yticks,fontsize = 18)
    ax1.set_xticks(Xticks)
    ax1.set_yticks(Yticks)
    ax1.set_ylim(Floor,Ceiling)
    ax1.set_xlabel('Participation coefficient, P',fontsize=20)
    ax1.set_ylabel('Within-module degree, Z',fontsize=20)

    ImageName = "../summary/" + "UserCarto_" + username + "_" + plot + ".pdf"
    fig.tight_layout()
    fig.savefig(ImageName, bbox_inches='tight',format='pdf',dpi = fig.dpi)


def everything():
    print "You have the following options:"
    print "1 - Analyze a hashtag"
    print "2 - Compare 2 hashtags"
    print "3 - Rank user in a hashtag"
    print "4 - Look for an individual user"
    print "5 - Merge one or more hashtags"

    opt=-1
    while opt not in range(6):
        opt= int(raw_input("Enter your option (0 to exit):\n"))
        #0)
        if opt==0:
            return
        #1)
        if opt==1:
            do_it()
        #2)
        if opt==2:
            check = raw_input("Is it OK to  consider the top 50 user? Y/N\n")
            if check[0] == 'Y' or check[0] == 'y':
                comparison()
            else:
                ntop = raw_input("Enter the number you want")
                comparison(ntop=int(ntop))
        #3)
        if opt==3:
            check = raw_input("Is it OK to  consider the top 20 user? Y/N\n")
            if check[0] == 'Y' or check[0] == 'y':
                users_ranking()
            else:
                ntop = raw_input("Enter the number you want")
                users_ranking(ntop=int(ntop))
        #4)
        if opt==4:
            one_user()

        #5)
        if opt==5:
            nh= int(raw_input("How many hashtags do you want to merge?"))
            hlist=[]
            name=''
            for i in range(nh):
                hlist+= [raw_input("Enter a hashatag (w/o #): ")]
                name+=str(hlist[i])+'_'
                G=join(hlist)
                node_attr= G.nodes(data=True)[0][1].keys()
                nxdigraph_to_gdf(G,str(hlist)+'.gdf',node_attr=node_attr)

        # None of the above:
        if opt not in range(6):
            print "Not an option."
    return
