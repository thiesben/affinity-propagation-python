# Imports
import matplotlib.pyplot as plt
import numpy as np
from typing import Any, Callable

def compute_responsibility(S:np.ndarray, R:np.ndarray, A:np.ndarray,
                           dampfac:float=0.5)->np.ndarray:
    '''Computes responsibilities
    Params:
        S: n by n matrix of similarities
        R: n by n matrix of current responsibilities
        A: n by n matrix of current availabilities
        dampfac: Damping factor used to calculate new responsibilities. Acts as weight
                 for the weighted sum of the old and new responsibilities.
    Returns:
        A matrix containing the new responsibilities.
    '''

    # Prepare the contents of the max
    to_max = A+S

    # "Empty" diagonal because it is not used in the max
    np.fill_diagonal(to_max,-np.inf)

    # Get row indices for subsetting the maximum values
    row_indices = np.arange(to_max.shape[0])

    # Get column indices of maximum values
    max_indices = np.argmax(to_max, axis=1)

    # Get maximum of each row
    row_max = to_max[row_indices, max_indices]
    # Assign -inf to previous maxima to get maxima without
    # "actual maxima"
    to_max[row_indices, max_indices] = -np.inf

    # Get secondary maximum of each row
    row_max_without = to_max[row_indices,
                                 np.argmax(to_max, axis=1)]

    # Create matrix of max(a(i, k') + s(i,k'))
    max_AS = np.zeros_like(S) + row_max.reshape(-1,1)

    # Modify values of those indices where there were maxima
    # because k' \neq k
    max_AS[row_indices, max_indices] = row_max_without

    return (1-dampfac) * (S - max_AS) + dampfac * R

def compute_availability(R:np.ndarray, A:np.ndarray,
                         dampfac:float=0.5)->np.ndarray:
    '''Computes availabilities
    Params:
        R: n by n matrix of current responsibilities
        A: n by n matrix of current availabilities
        dampfac: Damping factor used to calculate new availabilities. Acts as weight
                 for the weighted sum of the old and new availabilities.
    Returns:
        A matrix containing the new responsibilities.
    '''
    R = R.copy()
    Rdiag = np.diag(R).copy()
    np.fill_diagonal(R,0)  # Fill diagonal with 0
    R = np.where(R<0, 0, R) # Replace all negative responsibilities with 0

    # Compute availabilities

    # First, make matrix with column sum in each cell of that col
    # Note: This is still without diagonal / negative values
    Rsums = np.sum(R, axis=0)

    # a(i,k) = min(0, r(k,k) + sum(max(0,r(i',k)))
    new_A = np.minimum(  # min()
        0,  # 0
        Rdiag +  # r(k,k)
        Rsums - R  # sum(max(0,r(i', k)))
    )

    # Compute self-availabilities
    # Note that diagonal in R is 0
    np.fill_diagonal(new_A, np.sum(R, axis=0))
    return (1-dampfac)*new_A + dampfac*A

def sqeucl(u:np.ndarray, v:np.ndarray, axis:int=-1)->float:
    '''Given two vectors, calculates squared Euclidean Distance'''
    return(sqeucl_norm(u-v, axis=axis))

def sqeucl_norm(u, axis=-1):
    '''Calculates the squared euclidean norm of u along axis.'''
    return (u**2).sum(axis)

def neg_sqeucl(u:np.ndarray=None, v:np.ndarray=None,
               M:np.ndarray=None, axis:int=-1)->float:
    ''' Given two vectors, calculates negative squared Euclidean Distance.
        If fed with a matrix M, calculates pairwise row distances of M.
    Params:
        u, v: Pair of vectors to calculate distance for.
        M: optional, matrix to calculate pairwise row distances for.
        axis: Axis to calculate distance along (for broadcasting).
    Returns:
        A scalar for single vectors u, v. Otherwise a distance matrix.
    '''
    if M is not None:
        return -1 * (np.sum(M**2,axis=-1).reshape(-1,1) + \
                np.sum(M**2,axis=-1) - \
                2 * np.dot(M,M.T))
    else:
        if u is None or v is None:
            raise ValueError("If M is not given, both u and v have to given.")
        return(-sqeucl(u,v, axis=axis))

def cosim_one(u:np.ndarray, v:np.ndarray, axis:int=-1)->float:
    '''Given two vectors, calculates Cosine Similarity'''
    return (u@v)/(np.sqrt(u@u)*np.sqrt(v@v))

def cosim(u:np.ndarray=None, v:np.ndarray=None, M:np.ndarray=None):
    ''' Given two vectors, calculates cosine similarity.
        If fed with a matrix M, calculates pairwise row similarities of M.
    Params:
        u, v: Pair of vectors to calculate distance for.
        M: optional, matrix to calculate pairwise row distances for.
    Returns:
        A scalar for single vectors u, v. Otherwise a distance matrix.
    '''
    if M is not None:
        return np.dot(M,M.T)/(np.sqrt(np.sum(M**2,axis=-1).reshape(-1,1))*np.sqrt(np.sum(M**2,axis=-1)))
    else:
        if u is None or v is None:
            raise ValueError("If M is not given, both u and v have to given.")
        return(cosim_one(u,v))

def compute_similarity(M:np.ndarray,
                       func:Callable[[np.ndarray, np.ndarray], float]=neg_sqeucl,
                       measure="eucl",
                       u:np.ndarray=None, v:np.ndarray=None, axis:int=0)->np.ndarray:
    '''Computes negative euclidean distance (Similarities)
    Params:
        M: An n by p matrix to calculate similarities for
        func similarity measure
        u, v: Pair of vectors. If given, calculates similarity just for these two.
        axis: Treat either rows (0) as "observations" to calculate similarities for,
              or columns (1). If there are n observations with p features, similarities
              of observations will be calculated with axis=0 (default).
        measure: Only valid when M is given. Similarity measure for the similarity matrix.
                 Either "eucl" (default) for negative squared euclidean distance or "cos"
                 for cosine similarity.
    '''

    if M is None and (u is None or v is None):
        raise ValueError("Please specify either M or u AND v.")

    # If u and v are given, just calculate their similarity
    if u != None and v != None:
        return(func(u,v))

    # Else calculate similarity matrix for M
    if measure == "eucl":
        if axis == 0:
            return neg_sqeucl(M=M)
        if axis == 1:
            return neg_sqeucl(M=M.T)
    elif measure == "cos":
        if axis == 0:
            return cosim(M=M)
        if axis == 1:
            return cosim(M=M.T)
    else:
        raise ValueError(f"'measure' must either be 'eucl' or 'cos', not '{measure}'.")


def give_preferences(S:np.ndarray, preference:Any="median")->np.ndarray:
    '''Takes a similarity matrix and assigns equal "preferences" (values on the diagonal)
       according to the median of all similarities (excluding the diagonal).
    Params:
        S: Similarity matrix (n x n)
        preferences: Either "median", "min", a scalar numeric, or an n-dimensional np.ndarray
                     For details see README.
    '''
    indices = np.where(~np.eye(S.shape[0],dtype=bool))
    if preference == "median":
        m = np.median(S[indices])
    elif preference == "min":
        m = np.min(S[indices])
    elif type(preference) == np.ndarray:
        m = preference
    else:
        try:
            m = float(preference)
        except ValueError:
            raise ValueError("Parameter 'preference' must either be 'median', 'min', a np.ndarray or a scalar.")

    np.fill_diagonal(S, m)
    return S


def affinity_prop(X:np.ndarray, maxiter:int=100, preference:Any="median",
                  damping_factor:float=0.7, local_thresh:int=0,
                  message_thresh:int=0, calc_sim:bool=True,
                  sim_measure:str="eucl", calc_sim_axis:int=0, verbose:bool=True):
    '''Performs affinity propagation clustering.
    Params:
        X: Input matrix with data to cluster.
        maxiter: Maximum iterations after which to stop the clustering if it
                 does not converge before.
        preference: Either 'median' (default), 'min', a vector of the same size as the input data,
                    or a fixed scalar value. Determines the initial "preferences", i.e., self-similarities:
                    values on the diagonal of the similarity matrix. Details in the README.
        damping_factor: Damping factor used to calculate new availabilities and
                        responsibilities. Acts as weight for the weighted sum of
                        the old and new availabilities or responsibilities,
                        respectively. Lower values will lead to slower convergence, higher values will
                        prevent oscillation.
        local_thresh: Number of iterations without any change in the outcome labels before the algorithm stops.
        message_thresh: Threshold passed into np.allclose() used to stop the algorithm after messages
                        fall below that threshold.
        calc_sim: Whether or not to calculate a similarity matrix from the input data. Set to False if input
                  is already a similarity matrix (default: True).
        sim_measure: Similarity measure for the similarity matrix. Either "eucl" (default) for negative
                     squared euclidean distance or "cos" for cosine similarity.
        calc_sim_axis: Indicates whether similarity should be calculated between rows (0, default) or columns (1).
        verbose: Whether to be verbose.
    Returns:
        3-tuple of exemplars, labels (indicating the exemplar for every point),
        and centers (locations of the exemplars).
    '''

    X = np.asarray(X)

    # Convert input data into similarity matrix and add preferences
    if calc_sim:
        S = compute_similarity(X, axis=calc_sim_axis, measure=sim_measure)
    else:
        S = X
    S = give_preferences(S, preference=preference)

    # Initialize messages
    A = np.zeros_like(S)
    R = np.zeros_like(S)

    # Remove degeneracies to avoid oscillating
    S = S+1e-12*np.random.normal(size=A.shape) * (np.max(S)-np.min(S))

    # Initialize counting / indexing variables
    count_equal = 0
    i = 0
    converged = False

    if local_thresh == 0 and message_thresh == 0:
        while i < maxiter:
            R = compute_responsibility(S, R, A, dampfac=damping_factor)
            A = compute_availability(R, A, dampfac=damping_factor)
            i += 1

    else:
        # Stop if either maxiter is reached or local decisions stay constant
        while i < maxiter:
            E_old = R+A
            labels_old = np.argmax(E_old, axis=1)
            R = compute_responsibility(S, R, A, dampfac=damping_factor)
            A = compute_availability(R, A, dampfac=damping_factor)
            E_new = R+A
            labels_cur = np.argmax(E_new, axis=1)

            # Check if local decisions stayed constant
            if np.all(labels_cur == labels_old):
                count_equal += 1
            else:
                count_equal = 0

            # Check if messages changed
            if (message_thresh != 0 and np.allclose(E_old, E_new, atol=message_thresh)) or\
                (local_thresh != 0 and count_equal > local_thresh):
                converged = True
                break
            i += 1


    if verbose:
        if converged:
            print(f"Converged after {i} iterations.")
        else:
            print(f"Stopped after {maxiter} iterations.")

    E = R+A # Pseudomarginals
    labels = np.argmax(E, axis=1)
    exemplars = np.unique(labels)
    centers = X[exemplars]

    # Replace indices by increasing numbers
    replace = dict(zip(exemplars, range(len(exemplars))))
    mp = np.arange(0,max(labels)+1)
    mp[list(replace.keys())] = list(replace.values())
    labels = mp[labels]

    return exemplars, labels, centers

def cplot(data:np.ndarray, labels:np.ndarray, cmap:str="Set1")->None:
    '''Given two-dimensional data and cluster labels, plots data
       cluster assignment.
    Params:
        data: n by 2-dimensional array of input data
        labels: Array of cluster assignments of data
        cmap: Optional cmap for the plot (Default:"Set1")
    '''
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.scatter(data.T[0], data.T[1], c=labels, cmap=cmap)
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
    plt.show()
    return None
