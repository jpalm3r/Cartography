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

def DrawCarto(ax,Ceiling,Floor,hashtag, Percents = None, user = None, colorbox = None):

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
        fs = 20
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

    fcolor = "#FFE4A3"
    if colorbox != None:
        fcolor = colorbox

    for r in range(numQ):
        centerQX = Q[r][0][0] + Q[r][1]/2
        centerQY = Q[r][0][1] + Q[r][2]/2
        ax.add_patch(patches.Rectangle(Q[r][0], Q[r][1], Q[r][2], facecolor='white', edgecolor="Gray", ls = '--', lw = 0.7, alpha=0.1))
        if Percents != None:
            # Filling each quadrant proportional to the percentage
            ax.add_patch(patches.Rectangle(Q[r][0], Q[r][1], Percents[r]/100*Q[r][2], facecolor=fcolor, edgecolor="OrangeRed",lw = 0.0, alpha=0.7,zorder = 5))
            ax.text(centerQX, centerQY, Roles_Short[r] , horizontalalignment = 'center', verticalalignment = 'center',
                     color = color1, weight = "bold",fontsize= fs, style='normal', bbox={'facecolor':'white', 'alpha':0.0, 'pad':0},zorder = 10)
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

def UserBreakdown(username1, username2, username3 = None, plot='All'):

    import matplotlib.pyplot as plt
    import os
    import numpy as np
    import matplotlib.patches as mpatches

    # Script that takes 2 or 3 usernames and plot a comparison of their roles
    # for newtworks from different topics extracted from a gdf_gephi folder

    H = os.listdir('../gdf_gephi/')
    Hashtags = [h[:-4] for h in H]
    Hnums = range(1,len(H)+1)

    L = [str(topic[1]) + ' ' + topic[0] + '\n' for topic in zip(Hashtags, Hnums)]

    Lnew = [(word.replace('#','')).replace(' ', ' #') for word in L]
    legend = "".join(Lnew)
    patch = mpatches.Patch(color='None', label= legend)

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

    # Building the list of users
    USERS = [username1, username2]
    if username3 != None:
        USERS.append(username3)
    Nusers = len(USERS)
    # Each user have different values of Z,K and P in order to plot
    Ps = [[] for _ in range(Nusers)]
    Ks = [[] for _ in range(Nusers)]
    Zs = [[] for _ in range(Nusers)]
    Hs = [[] for _ in range(Nusers)]
    Roles = [[0,0,0,0,0,0,0] for _ in range(Nusers)]
    P = []
    T = [0 for _ in range(Nusers)]

    for i in range(Nusers):
        username = USERS[i]
        # Look for a GDF file
        for topic in Hashtags:
            G = gdf_to_nxdigraph(topic)
            if username in G.nodes():
                user_role = G.node[username][role]
                index = int(user_role[1:]) - 1
                Roles[i][index] += 1
                Ps[i].append(float(G.node[username]['P_coef']))
                Zs[i].append(float(G.node[username][zKind]))
                Hs[i].append(topic)
                Ks[i].append(G.degree(username))
                T[i] += 1

        P.append([float(r)*100/T[i] for r in Roles[i]])

    KsFlat = [item for sublist in Ks for item in sublist]
    ZsFlat = [item for sublist in Zs for item in sublist]

    kMax = max(KsFlat)
    KsNorm = [[float(k)/kMax for k in K] for K in Ks]
    KsCale = [[2000*np.sqrt(x/(x+0.1)) for x in K] for K in KsNorm]

    # Creating main figure. It will contain the scatterplot
    fig, ax1 = plt.subplots(1,1, figsize = (20,12))
    # Creating subfigures, thye contain the box diagram with percentages
    h_ax = w_ax = 0.45
    y_ax = -0.48
    ax2 = fig.add_axes([0.04, y_ax, w_ax, h_ax]) # [relX, relY, relWidth, relHeight]
    ax3 = fig.add_axes([0.53, y_ax, w_ax, h_ax]) # [relX, relY, relWidth, relHeight]
    for i in range(Nusers):
        for label, x, y in zip(Hs[i],Ps[i],Zs[i]):
                indInH = Hashtags.index(label)
                ax1.annotate(Hnums[indInH], xy=(x, y), xytext=(x, y - 0.1),
                #                     arrowprops=dict(arrowstyle="->", connectionstyle="arc3"),
                             textcoords='data',fontsize = 16,zorder=150)


    Ceiling = int(max(ZsFlat)) + 3
    Floor = int(min(ZsFlat,-3))

    c1 = c2 = 8
    f1 = f2 = -2

    color1 = '#F26267'
    edgecolor1 = 'Crimson'
    color2 = '#0086b3'
    edgecolor2 = '#003399'

    ax1 = DrawCarto(ax1,Ceiling,Floor,topic)
    ax2 = DrawCarto(ax2,c1,f1,topic,Percents=P[0],user=username1,colorbox=color1)
    ax3 = DrawCarto(ax3,c2,f2,topic,Percents=P[1],user=username2,colorbox=color2)

    Ceiling += 3 # This happens in DrawCarto
    ax1.scatter(Ps[0],Zs[0],s = KsCale[0], c= color1, edgecolor = edgecolor1, lw = 1.0, zorder=100)
    ax1.scatter(Ps[1],Zs[1],s = KsCale[1], c= color2, edgecolor = edgecolor2, lw = 1.0, zorder=100)

    #   WIP WIP WIP
    #    if username3 != None:
    #        ax1.scatter(Ps[2],Zs[2],s = KsCale[2], c= '#e600ac', edgecolor = '#b300b3', lw = 1.0, zorder=100)

    topicLegend = ax1.legend(handles=[patch],labels = [legend], title="Topics",fontsize=18,frameon=False, ncol=1,loc="upper left",bbox_to_anchor=(0.98,1.0))
    plt.setp(topicLegend.get_title(),fontsize=22)
    ax2.text(0.5, (c1 + f1)/2, '@' + username1, zorder = 30, horizontalalignment = 'center', verticalalignment = 'center',
             color = edgecolor1, weight = "bold",fontsize= 42, style='normal', bbox={'facecolor':'white', 'alpha':0.0, 'pad':8})
    ax3.text(0.5, (c2 + f2)/2, '@' + username2, zorder = 30, horizontalalignment = 'center', verticalalignment = 'center',
             color = edgecolor2, weight = "bold",fontsize= 42, style='normal', bbox={'facecolor':'white', 'alpha':0.0, 'pad':8})

    ax1.set_title('User Breakdown (' + plot + ' degree)',fontsize = 32)
    Xticks = [0.0,0.2,0.4,0.6,0.8,1.0]
    nYticks = (Ceiling - Floor)/5
    Yticks = range(Floor,Ceiling)[::nYticks]
    ax1.set_xticklabels(Xticks,fontsize = 20)
    ax1.set_yticklabels(Yticks,fontsize = 20)
    ax1.set_xticks(Xticks)
    ax1.set_yticks(Yticks)
    ax1.set_ylim(Floor,Ceiling)
    ax1.set_xlabel('Participation coefficient, P',fontsize=28)
    ax1.set_ylabel('Within-module degree, Z',fontsize=28)

    ImageName = "../summary/" + "UserCarto_" + username1 + "_vs_" + username2 +  "_" + plot + ".pdf"
    fig.tight_layout()
    fig.savefig(ImageName, bbox_inches='tight',format='pdf',dpi = fig.dpi)
