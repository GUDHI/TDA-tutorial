def hausd_interval(data, level = 0.95, m=-1, B =1000,pairwise_dist = False,
                   leaf_size = 2,ncores = None):
    
    '''
    Subsampling Confidence Interval for the Hausdorff Distance between a 
    Manifold and a Sample
    Fasy et al AOS 2014
    
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







def truncated_simplex_tree(st,int_trunc=100):
    '''
    This function return a truncated simplex tree 
    
    Input:
    st : a simplex tree
    int_trunc : number of persistent interval keept per dimension (the largest)
    
    Ouptut:
    st_trunc_pers : truncated simplex tree    
    '''
    
    st.persistence()    
    dim = st.dimension()
    st_trunc_pers = [];
    for d in range(dim):
        pers_d = st.persistence_intervals_in_dimension(d)
        d_l= len(pers_d)
        if d_l > int_trunc:
            pers_d_trunc = [pers_d[i] for i in range(d_l-int_trunc,d_l)]
        else:
            pers_d_trunc = pers_d
        st_trunc_pers = st_trunc_pers + [(d,(l[0],l[1])) for l in pers_d_trunc]
    return(st_trunc_pers)


