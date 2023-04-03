import numpy as np

### metrics
def ssd(A, B):
    return np.sum(np.power(A-B, 2))

def mad(A,B):
    return np.max(np.abs(A-B))

def sad(A,B):
    return np.sum(np.abs(A,B))

### subroutines for algo
def blockmeasure(A, B, measure):
    # computes a measure between each block
    if measure == 'ssd':
        out = ssd(A,B)
    elif measure == 'mad':
        out = mad(A,B)
    elif measure == 'sad':
        out = sad(A,B)
    else
        raise Exception('Unknown metric: '+str(measure))

def getblockindices(i, j, d0q, d1q, blocksize, framesize):
    # Inputs:
    # i     = starting index of ref block along dim 0
    # j     = starting index of ref block along dim 1
    # d0q   = Query displacements along dim 0
    # d1q   = Query displacements along dim 1
    # blocksize = ...
    # framesize = ...
    # 
    # Returns:
    # List of block-sets ~ enables unequal block sizes on edges
    # All block in a block-set have the same size ~ enables parallelism
    # Block measure functions can be called optimally on each block-set

    P = np.zeros((2,1))
    P[0,0] = i 
    P[1,0] = j

    dq = np.vstack(d0q, d1q)

    # For each query point in dq, we will define the following:
    ref_start   = np.tile(P, (1, dq.shape[1]))
    ref_end     = ref_start + block_size
    block_start = P + dq
    block_end   = block_start + block_size

    # q0 = i+d0q      # 5 + [-3 -2 -1 -3 -2 -1 -3 -2 -1]
    # q1 = j+d1q      # 3 + [ 2  2  2  1  1  1  0  0  0]

    ## Discard any impossible search idices
    # block_start of some queries can sometimes take invalid values larger than frame size - discard them
    tfs         = block_start < framesize
    idx         = tfs[0,:] and tfs[1,:]
    ref_start   = ref_start[:,idx]
    ref_end     = ref_end[:,idx]
    block_start = block_start[:,idx]
    block_end   = block_end[:,idx]
    # block_end of some queries can sometimes take invalid values smaller than frame size - discard them
    tfs         = block_end > 0
    idx         = tfs[0,:] and tfs[1,:]
    ref_start   = ref_start[:,idx]
    ref_end     = ref_end[:,idx]
    block_start = block_start[:,idx]
    block_end   = block_end[:,idx]
    # block_start of some queries can sometimes take partially invalid negative values - round them carefully
    tfs                 = block_start < 0
    idx                 = tfs[0,:] or tfs[1,:]
    ref_start[:,idx]    = ref_start[:,idx] - block_start[:,idx]
    ref_end             = ref_end
    block_start[:,idx]  = 0
    block_end           = block_end
    # block_end of some queries can sometimes take partially invalid negative values - round them carefully
    tfs                 = block_end > framesize
    idx                 = tfs[0,:] or tfs[1,:]
    ref_start           = ref_start
    ref_end[:,idx]      = ref_end[:,idx] - (block_end[:,idx] - framesize)
    block_start         = block_start
    block_end[:,idx]    = framesize

    ## Create block-sets for edge cases
    # Compute dimensions of each query block
    ref_size    = ref_end   - ref_start
    block_size  = block_end - block_start
    assert np.all(ref_size == block_size)
    # Group based on sizes
    block_sets = []
    for x in np.unique(block_size, axis=0):
        idx = block_size == x
        block_set.ref_size   = ref_size[idx]
        block_set.block_size = block_size[idx]
        block_sets.append(block_set)

    return block_sets

def compute(frame1, frame2, i, j, d0q, d1q, blocksize, metric):
    
    # get indices for all block matches to check for
    block_sets = getblockindices(i, j, d0q, d1q, blocksize, frame1.shape)

    # iterate over each block-set
    for block_set in block_sets:
        block_set = block_sets[i]

    # compute metric

def getmetrics(frame1, frame2, i, j, blocksize, searchwindow, metric, metrics):
    
    # indices
    D0 = np.repeat(np.arange(searchwindow[0]), searchwindow[1]) #  0  0  0  1  1  1  2  2  2
    D1 = np.tile(np.arange(searchwindow[1]), searchwindow[0])   #  0  1  2  0  1  2  0  1  2
    idx = D0*searchwindow[0]+D1                                 #  0  1  2  3  4  5  6  7  8

    d0 = D0 - searchwindow[0]//2                                # -1 -1 -1  0  0  0  1  1  1
    d1 = D1 - searchwindow[1]//2                                # -1  0  1 -1  0  1 -1  0  1
    
    # dummy variable to store outputs
    localmetrics = np.ones(np.product(searchwindow)) * -1

    # query values in "memory"
    vals = metrics[i+d0, j+d1, :]
    existing = np.all(vals == -1, axis=1)

    # fetch existing values
    localmetrics[idx[existing]] = vals[idx]

    # compute non existing values
    d0q = d0[idx[not existing]]
    d0q = d1[idx[not existing]]
    localmetrics[idx[not existing]] = compute(frame1, frame2, i, j, d0q, d0q, blocksize, metric)    

    return localmetrics


### Algo
def fsalgo(frame1, frame2, blocksize, searchwindow, metric):
    # overall computation of full-search and returns d0,d1 estimates 
    # uses subroutines to split tasks
    # uses memoization to avoid redoing computation

    assert frame.shape == frame2.shape, "Frames are not of the same shape"
    assert np.count_nonzero(blocksize.shape)==2, "Block size should be a tuple with 2 elements"

    # collect metrics for each displacement pair in the following variable
    # initialize to -1 since none of the metrics considered can take negative values
    metrics = np.ones(frame1.shape+(np.product(blocksize),), dtype=float) * -1

    for i in range(frame1.shape[0]):
        for j in range(frame1.shape[1]):
            metrics[i,j,:] = getmetrics(frame1, frame2, i, j, blocksize, searchwindow, metric, metrics)

    # find optimal d0, d1 estimates from evaluated distances in search space
    optimal = np.argmin(metrics, axis=2)
    optimal_d0 = optimal // searchwindow[0] - searchwindow[0] // 2 # check this
    optimal_d1 = optimal %  searchwindow[0] - searchwindow[1] // 2 # check this
    optimaldisp = np.stack(optimal_d0, optimal_d1, axis=2)

    return optimaldisp

### Main
def main():
    # loads frames and runs full-search algo on them

    # hyper-params
    blocksize = [16, 16]
    searchwindow = [48, 48]
    metric = 'sad'
    
    # load frames
    
    # call algo
    dispest = fsalgo(frame1, frame2, blocksize, searchwindow, metric)

    # compare with ground truth
