def hausd_interval(data, level = 0.95, m=-1, B =1000,pairwise_dist = False,
                   leaf_size = 2,ncores = None):
    
    '''
    Subsampling Confidence Interval for the Hausdorff Distance between a 
    Manifold and a Sample
    Ref: Fasy et al,CONFIDENCE SETS FOR PERSISTENCE DIAGRAMS,  AOS 2014
    
    Input:
    data : a nxd numpy array representing n points in R^d, or a nxn matrix of pairwise distances
    m : size of each subsample. If m=-1 then m = n / np.log(n)   
    B : number of subsamples 
    level : confidence level
    pairwise_dist : if pairwise_dist = True then data is a nxn matrix of pairwise distances
    leaf_size : leaf size for KDTree
    ncores :  number of cores for multiprocessing (if None then the maximum number of cores is used)
        
    Output: 
    quantile for the Hausdorff distance
        
   
    '''
    
    
    import numpy as np
    from multiprocessing import Pool
    from sklearn.neighbors import KDTree

    
    # sample size
    n = np.size(data,0)
    
    # subsample size
    if m == -1:
        m = int (n / np.log(n))
    
    
    # Data is an array
    if pairwise_dist == False:
            
        # for subsampling
        # a reprendre sans shuffle slit   

        
        global hauss_dist
        def hauss_dist(m):
            '''
            Distances between the points of data and a random subsample of data of size m
            '''            
            I = np.random.choice(n,m)
            Icomp = [item for item in np.arange(n) if item not in I]
            tree = KDTree(data[I,],leaf_size=leaf_size)
            dist, ind = tree.query(data[Icomp,],k=1) 
            hdist = max(dist)
            return(hdist)
        
        # parrallel computing
        with Pool(ncores) as p:
            dist_vec = p.map(hauss_dist,[m]*B)
        p.close()
        dist_vec = [a[0] for a in dist_vec]        
          
        
    # Data is a matrix of pairwise distances    
    else:
        def hauss_dist(m):
            '''
            Distances between the points of data and a random subsample of data of size m
            '''
            I = np.random.choice(n,m)    
            hdist= np.max([np.min(data[I,j]) for j in np.arange(n) if j not in I])              
            return(hdist)
            
        # parrallel computing
        with Pool(ncores) as p:
            dist_vec = p.map(hauss_dist, [m]*B)
        p.close()
    
            
    # quantile and confidence band
    myquantile = np.quantile(dist_vec, level)
    c = 2 * myquantile     
            
    return(c)


def sort_thresh_pers(barcode,int_trunc = None, pers_length = None):
    '''
    This function returns  ..........
    
    Input:
    barcode
    int_trunc : number of persistent intervals to keep
    
    Ouptut:
     
    '''
    
    import numpy as np   

    # sort the simplex tree
    long = [b[1][1] - b[1][0] for b in barcode]
    I  = np.argsort(long)[::-1]
    barcode_sort =  [barcode[i] for i in I]

        
    if int_trunc != None:
        
        barcode_sort = barcode_sort[::int_trunc]        
       
    if pers_length != None:
        barcode_sort = [b for b in barcode_sort if b[1][1] - b[1][0] > pers_length]
        
    return(barcode_sort)





def bottleneck_bootstrap_data_points(data,filt = "Rips",level = 0.95,B= 100,
                                     pairwise_dist = False,max_edge_length = None,
                                     max_dimension=1,metric = None,ncores = None,
                                     output_dvec = False):
    '''
    Bottleneck bootstrap method for confidence sets of persistence diagrams
    for data filtrations built on data points
    Ref: 
    
    
    Input:
    data : a nxd numpy array representing n points in R^d, or a nxn matrix of pairwise distances  
    filt : filtration type : Rips or Alpha complex
    B : number of subsamples 
    level : confidence level
    pairwise_dist : if pairwise_dist = True then data is a nxn matrix of pairwise distances
    max_edge_length = max_edge_length for rips skeleton
    metric : metric on the space of persistance diagrams
    ncores :  number of cores for multiprocessing (if None then the maximum number of cores is used)
    max_dimension : max dimension of topological features ( = max dimension-1 of simplices  in simplex trees)
    output_dvec : output_dvec = True returns the vector of boostrap bootlneck distances
        
    Output: 
    st_out : simplex tree for the data
    quantile for the distribution of  metric(st_out,st_boot) where st_sample is the simplex tree of a bootstrap sample
 
    '''
    from multiprocessing import Pool    
    import numpy as np
    import gudhi as gd 

    global bottle_map
    
    if metric == None:
        metric  = gd.bottleneck_distance
    

    if filt == "Alpha":
        n,d = data.shape   
        alpha_complex = gd.AlphaComplex(points=data)
        st_alpha  = alpha_complex.create_simplex_tree()
        st_out = st_alpha.persistence()
        BarCodes_list = [st_alpha.persistence_intervals_in_dimension(dim) for dim in np.arange(max_dimension+1)]
        
        def bottle_map(md):
            I_b = np.random.choice(n,n)
            alpha_complex_b = gd.AlphaComplex(points=data[I_b,:])
            st_alpha_b  = alpha_complex_b.create_simplex_tree()
            st_alpha_b.persistence();
            bot_b = 0
            for dim in np.arange(md+1):
                interv_b_dim =  st_alpha_b.persistence_intervals_in_dimension(dim)
                bot_b  = max(bot_b,metric(BarCodes_list[dim],interv_b_dim))
            return(bot_b)


    if (filt == "Rips") & (pairwise_dist == False) :
  
        n,d = data.shape
        skeleton  = gd.RipsComplex(points = data ,max_edge_length=max_edge_length)
        # simplices of dim max_dimension+1 to infer topological features of dim max_dimension 
        rips_simplex_tree = skeleton.create_simplex_tree(max_dimension=max_dimension+1)
        st_out = rips_simplex_tree.persistence()
        BarCodes_list = [rips_simplex_tree.persistence_intervals_in_dimension(dim) for dim in np.arange(max_dimension+1)]
        
        def bottle_map(md):
            I_b = np.random.choice(n,n)
            skeleton_b  = gd.RipsComplex(points=data[I_b,:],max_edge_length=max_edge_length)
            rips_simplex_tree_b = skeleton_b.create_simplex_tree(max_dimension=max_dimension+1)
            rips_simplex_tree_b.persistence();
            bot_b = 0
            for dim in np.arange(md+1):
                interv_b_dim =  rips_simplex_tree_b.persistence_intervals_in_dimension(dim)
                bot_b  = max(bot_b,metric(BarCodes_list[dim],interv_b_dim))
            return(bot_b)


    if (filt == "Rips") & (pairwise_dist == True) :
  
        n,d = data.shape
        skeleton  = gd.RipsComplex(distance_matrix=data,max_edge_length=max_edge_length)
        # simplices of dim max_dimension+1 to infer topological features of dim max_dimension
        rips_simplex_tree = skeleton.create_simplex_tree(max_dimension=max_dimension+1)
        st_out = rips_simplex_tree.persistence()
        BarCodes_list = [rips_simplex_tree.persistence_intervals_in_dimension(dim) for dim in np.arange(max_dimension+1)]
        
        def bottle_map(md):
            I_b = np.random.choice(n,n)
            skeleton_b  = gd.RipsComplex(distance_matrix=data[I_b,:][:,I_b],max_edge_length=max_edge_length)
            rips_simplex_tree_b = skeleton_b.create_simplex_tree(max_dimension=max_dimension+1)
            rips_simplex_tree_b.persistence();
            bot_b = 0
            for dim in np.arange(md+1):
                interv_b_dim =  rips_simplex_tree_b.persistence_intervals_in_dimension(dim)
                bot_b  = max(bot_b,metric(BarCodes_list[dim],interv_b_dim))
            return(bot_b)




    with Pool(ncores) as p:
        dist_vec = p.map(bottle_map,[max_dimension]*B)
    p.close()                 
    myquantile = np.quantile(dist_vec, level)        
    
    if output_dvec == True:
        return (myquantile,st_out,dist_vec)
    else:
        return (myquantile,st_out)





def bootstrap_band_function(data,hatf,grid,
                           level = 0.95,B= 100,
                           max_dimension=1,metric = None,                           
                           ncores = None,output_dvec = False):
    '''
    bootstrap l_infinity band for confidence sets of persistence diagrams
    
    Ref: Fasy BT, Lecci F, Rinaldo A, Wasserman L, Balakrishnan S, Singh A (2014). “Confidence sets for persistence diagrams.” Ann. Statist., 42(6), 2301–2339
    Ref: Frédéric Chazal, Brittany Fasy, Fabrizio Lecci, Bertrand Michel, Alessandro Rinaldo, Alessandro Rinaldo, Larry Wasserman (2017)
    Robust topological inference: Distance to a measure and kernel distance. The Journal of Machine Learning Research 18 (1), 5845-5884.
 
    
    Input:
    data : a nxd numpy array representing n points in R^d  
    hatf  : hatf(X,x) : a data dependant function, as for instance a density estimator. X: observations : x : querry dataset
    grid : grid for the cubical complex
    level : confidence level
    B : number of subsamples 
    ncores :  number of cores for multiprocessing (if None then the maximum number of cores is used)
    output_dvec : output_dvec = True returns the vector of boostrap bootlneck distances
   
       
    Output: 
    quantile for the distribution of ....
 
    '''
    from multiprocessing import Pool    
    import numpy as np

    global boot_map
    
    n,p = data.shape
    
    values = hatf(data,grid)
    
    # TODO : find how to deal with multiprocessing and no arguments in bottle_map
    def boot_map(u):
        I_b = np.random.choice(n,n)
        data_b = data[I_b,:]
        values_b = hatf(data_b,grid)
        return(np.max(np.abs(values_b - values)))
    
    with Pool(ncores) as p:
        dist_vec = p.map(boot_map,[0]*B)
    p.close()
    
    myquantile = np.quantile(np.array(dist_vec), level)
    
    if output_dvec == True:
        return (myquantile,dist_vec)
    else:
        return (myquantile)



def bottleneck_bootstrap_function(data,hatf,grid,metric = None,
                                  level = 0.95,B= 100,ncores = None,
                                  max_dimension=1,
                                  output_dvec = False):
    '''
    Bottleneck bootstrap for persistence diagrams of filtrations of sublevel sets of functions
    
    Ref: Frédéric Chazal, Brittany Fasy, Fabrizio Lecci, Bertrand Michel, Alessandro Rinaldo, Alessandro Rinaldo, Larry Wasserman (2017)
    Robust topological inference: Distance to a measure and kernel distance. The Journal of Machine Learning Research 18 (1), 5845-5884.
 
    
    Input:
    data : a nxd numpy array representing n points in R^d  
    hatf  : hatf(X,x) : a data dependant function, as for instance a density estimator. X: observations : x : querry dataset
    grid : grid for the cubical complex
    metric : metric on the space of persistance diagrams
    max_dimension : max dimension of topological features ( = max dimension of simplices -1  in simplex trees)
    B : number of subsamples 
    level : confidence level  
    ncores :  number of cores for multiprocessing (if None then the maximum number of cores is used)
    output_dvec : output_dvec = True returns the vector of boostrap bootlneck distances
       
    Output: 
    quantile for the distribution of ....
 
    '''
    
    
    
    from multiprocessing import Pool    
    import numpy as np
    import gudhi as gd     

    
    if metric == None:
        metric  = gd.bottleneck_distance  

    global boot_map
    
    n,p = data.shape
    nx,ny = grid.shape
    

     # cubical complex of the data     
    values = hatf(data,grid)
    cc_data= gd.CubicalComplex(dimensions= [nx,ny],
                               top_dimensional_cells = hatf(data,grid))
    pers_data = cc_data.persistence()
    BarCodes_list = [pers_data.persistence_intervals_in_dimension(dim) for dim in np.arange(max_dimension+1)]
    
    
    def bottle_map(i):
        print(i)
        I_b = np.random.choice(n,n)
        cc_b= gd.CubicalComplex(dimensions= [nx,ny],
                                top_dimensional_cells = hatf(data[I_b,:],grid))
        pers_b = cc_b.persistence()
        bot_b = 0
        for dim in np.arange(max_dimension+1):
            interv_b_dim =  pers_b.persistence_intervals_in_dimension(dim)
            bot_b  = max(bot_b,metric(BarCodes_list[dim],interv_b_dim))
        return(bot_b)
        

    with Pool(ncores) as p:
        dist_vec = p.map(bottle_map,np.arange(B))
    p.close()                 
    myquantile = np.quantile(dist_vec, level)        
    
    if output_dvec == True:
        return (myquantile,pers_data,dist_vec)
    else:
        return (myquantile,pers_data)



       
