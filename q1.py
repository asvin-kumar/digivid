import numpy as np
from skimage import io
import scipy.io
import traceback

class blockset:
    ref_start:  int
    ref_end:    int
    blk_start:  int
    blk_end:    int
    idx:        bool
    blk_size:   int

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
    else:
        raise Exception('Unknown metric: '+str(measure))
    
    return out

def aand(A, B):
    return np.logical_and(A, B)

def aor(A, B):
    return np.logical_or(A, B)

def anot(A):
    return np.logical_not(A)

def getblockindices(i, j, d0q, d1q, blk_size, framesize):
    # Inputs:
    # i     = starting index of ref block along dim 0
    # j     = starting index of ref block along dim 1
    # d0q   = Query displacements along dim 0
    # d1q   = Query displacements along dim 1
    # blocksize = ...
    # framesize = ...
    # 
    # Returns:
    # List of block-sets: enables unequal block sizes on edges
    # All block in a block-set have the same size: enables parallelism
    # Block measure functions can be called optimally on each block-set

    P = np.zeros((2,1), dtype=int)
    P[0,0] = i 
    P[1,0] = j

    dq = np.vstack((d0q, d1q))

    # some required variables
    rs = P
    re = np.minimum(P+blk_size, framesize)
    bs = re-rs

    # For each query point in dq, we will define the following:
    ref_start   = np.tile(P, (1, dq.shape[1]))
    ref_end     = ref_start + bs
    blk_start   = P + dq
    blk_end     = blk_start + bs

    valid       = np.ones(dq.shape[1], dtype=bool)

    # q0 = i+d0q      # 5 + [-3 -2 -1 -3 -2 -1 -3 -2 -1]
    # q1 = j+d1q      # 3 + [ 2  2  2  1  1  1  0  0  0]

    # ## Discard any impossible search idices
    # # blk_start of some queries can sometimes take invalid values larger than frame size - discard them
    # tfs             = blk_start < framesize
    # idx             = aand(tfs[0,:], tfs[1,:])
    # ref_start       = ref_start[:,idx]
    # ref_end         = ref_end[:,idx]
    # blk_start       = blk_start[:,idx]
    # blk_end         = blk_end[:,idx]
    # dq              = dq[:,idx]
    # valid[anot(idx)]  = False
    # # blk_end of some queries can sometimes take invalid values smaller than frame size - discard them
    # tfs             = blk_end > 0
    # idx             = aand(tfs[0,:], tfs[1,:])
    # ref_start       = ref_start[:,idx]
    # ref_end         = ref_end[:,idx]
    # blk_start       = blk_start[:,idx]
    # blk_end         = blk_end[:,idx]
    # dq              = dq[:,idx]
    # valid[anot(idx)]  = False

    # blk_start of some queries can sometimes take partially invalid negative values - round them carefully
    tfs                     = blk_start < 0

    idx                     = tfs[0,:]
    if np.any(idx):
        ref_start[0,idx]    = ref_start[0,idx] - blk_start[0,idx]
        ref_end             = ref_end
        blk_start[0,idx]    = 0
        blk_end             = blk_end

    idx                     = tfs[1,:]
    if np.any(idx):
        ref_start[1,idx]    = ref_start[1,idx] - blk_start[1,idx]
        ref_end             = ref_end
        blk_start[1,idx]    = 0
        blk_end             = blk_end

    # blk_end of some queries can sometimes take partially invalid negative values - round them carefully
    tfs                     = blk_end > framesize

    idx                     = tfs[0,:]
    if np.any(idx):
        ref_start           = ref_start
        ref_end[0,idx]      = ref_end[0,idx] - (blk_end[0,idx] - framesize[0])
        blk_start           = blk_start
        blk_end[0,idx]      = framesize[0]

    idx                     = tfs[1,:]
    if np.any(idx):
        ref_start           = ref_start
        ref_end[1,idx]      = ref_end[1,idx] - (blk_end[1,idx] - framesize[1])
        blk_start           = blk_start
        blk_end[1,idx]      = framesize[1]

    ## Create block-sets for edge cases
    # Compute dimensions of each query block
    ref_sizes    = ref_end   - ref_start
    blk_sizes    = blk_end   - blk_start
    assert np.all(ref_sizes == blk_sizes)
    # Group based on sizes
    blk_sets = []
    uniques = np.unique(blk_sizes, axis=1)
    for u in range(uniques.shape[1]):
        x                   = uniques[:,u].reshape(2,1)
        idx                 = np.all(blk_sizes==x, axis=0)
        blk_set             = blockset()
        blk_set.ref_start   = ref_start[:,idx]
        blk_set.ref_end     = ref_end[:,idx]
        blk_set.blk_start   = blk_start[:,idx]
        blk_set.blk_end     = blk_end[:,idx]
        blk_set.idx         = idx
        blk_set.blk_size    = x
        blk_sets.append(blk_set)

    return blk_sets, valid

def compute(frame1, frame2, i, j, d0q, d1q, blocksize, metric):
    
    framesize = np.array(frame1.shape).reshape(2,1)

    vals = np.ones(d0q.shape) * -1

    # get indices for all block matches to check for
    blk_sets, valid = getblockindices(i, j, d0q, d1q, blocksize, framesize)

    # iterate over each block-set
    for bs in blk_sets:
        # Setup blocks to measure distance between
        dim0 = int(bs.blk_size[0])
        dim1 = int(bs.blk_size[1])
        dim2 = int(bs.ref_start.shape[1])
        A = np.zeros((dim0, dim1, dim2))
        B = np.zeros((dim0, dim1, dim2))
        # assert np.all(bs.ref_start==bs.ref_start[:,0])
        # A = np.zeros((dim0, dim1, 1))
        # A = frame1[bs.ref_start[0,j]:bs.ref_end[0,j], bs.ref_start[1,j]:bs.ref_end[1,j]]
        try:
            for j in range(bs.ref_start.shape[1]):
                A[:,:,j] = frame1[bs.ref_start[0,j]:bs.ref_end[0,j], bs.ref_start[1,j]:bs.ref_end[1,j]]
                B[:,:,j] = frame2[bs.blk_start[0,j]:bs.blk_end[0,j], bs.blk_start[1,j]:bs.blk_end[1,j]]
        except Exception:
            traceback.print_exc()
            print(A.shape, B.shape, \
                  frame1[bs.ref_start[0,j]:bs.ref_end[0,j], bs.ref_start[1,j]:bs.ref_end[1,j]].shape, \
                  frame2[bs.blk_start[0,j]:bs.blk_end[0,j], bs.blk_start[1,j]:bs.blk_end[1,j]].shape, \
                  sep='\n',end='\n@@@\n')
            print(j, bs.blk_size, frame1.shape, frame2.shape, \
                  bs.ref_start, bs.ref_end, bs.blk_start, bs.blk_end, \
                  bs.ref_start.shape, bs.ref_end.shape, bs.blk_start.shape, bs.blk_end.shape, \
                    sep='\n', end='\n***\n')
            raise Exception
        # compute metric
        met = blockmeasure(A, B, metric)
        vals[bs.idx] = met

    return vals

def getmetrics(frame1, frame2, i, j, blocksize, searchwindow, metric, metrics):
    
    framesize = frame1.shape

    # indices
    D0 = np.repeat(np.arange(searchwindow[0]), searchwindow[1]) #  0  0  0  1  1  1  2  2  2
    D1 = np.tile(np.arange(searchwindow[1]), searchwindow[0])   #  0  1  2  0  1  2  0  1  2

    d0 = D0 - searchwindow[0]//2                                # -1 -1 -1  0  0  0  1  1  1
    d1 = D1 - searchwindow[1]//2                                # -1  0  1 -1  0  1 -1  0  1
    
    # dummy variable to store outputs
    localmetrics = np.ones(np.product(searchwindow)) * -1

    '''
    sw 4
    0 1 2 3
    (-2) -2 -1 0 1
    (*-1) 2 1 0 -1
    (+1~2) 3 2 1 0
    sw 5
    0 1 2 3 4 
    (-2) -2 -1 0 1 2
    (*-1) 2 1 0 -1 -2
    (+2=2) 4 3 2 1 0

    '''

    # query values in "memory"
    # these are the z values at the x,y query locations where our metric 
    # with current ref block are stored.
    # Needs a -1 because ranges of -d0 or -d1 are different from d0 and d1
    D0q = -d0 + searchwindow[0]//2 - 1 + searchwindow[0]%2
    D1q = -d1 + searchwindow[1]//2 - 1 + searchwindow[1]%2

    x = i+d0
    y = j+d1
    z = D0q * searchwindow[0] + D1q
    
    # We have four types of query locations
    # 1. Invalid by either being <=-blocksize or >=framesize
    # 2. Valid but partials hence absent
    # 3. Valid but already computed
    # 4. Valid but not yet computed
    # Here's how we're going to deal with them
    # Case 1: Set metric to -1
    # Case 2: Compute metric
    # Case 3: Fetch computed metric
    # Case 4: Compute metric

    # Case 1 - Invalid
    valid = np.ones(np.product(searchwindow), dtype=bool)
    # Sometimes x can be less than -blocksize[0]
    tfs             = x>-blocksize[0]
    valid[anot(tfs)]  = False
    # Sometimes x can be more than framesize[0]
    tfs = x<framesize[0]
    valid[anot(tfs)]  = False
    # Sometimes y can be less than -blocksize[1]
    tfs             = y>-blocksize[1]
    valid[anot(tfs)]  = False
    # Sometimes y can be more than framesize[1]
    tfs = y<framesize[1]
    valid[anot(tfs)]  = False

    # Case 2 - It's valid but it's still absent because it's a partial
    partial = aand(aor(x<0, y<0) , valid)
    d0q = d0[partial]
    d1q = d1[partial]
    localmetrics[partial] = compute(frame1, frame2, i, j, d0q, d1q, blocksize, metric)

    # Case 3 - It's valid and it's not a partial overlap so it may have been computed already
    sel = aand(valid, anot(partial))
    vals = np.ones(np.product(searchwindow), dtype=bool) * True
    vals[anot(sel)] = -1
    vals[sel] = metrics[x[sel], y[sel], z[sel]]
    present = vals != -1
    localmetrics[present] = vals[present]

    # Case 4 - It's valid and it's not a partial and it's not present so it needs to be computed
    sel2 = aand(aand(valid, anot(partial)), anot(present))
    d0q = d0[sel2]
    d1q = d1[sel2]
    localmetrics[sel2] = compute(frame1, frame2, i, j, d0q, d1q, blocksize, metric)

    return localmetrics


### Algo
def fsalgo(frame1, frame2, blocksize, searchwindow, metric):
    # overall computation of full-search and returns d0,d1 estimates 
    # uses subroutines to split tasks
    # uses memoization to avoid redoing computation

    assert frame1.shape == frame2.shape, "Frames are not of the same shape"
    assert np.count_nonzero(blocksize.shape)==2, "Block size should be a tuple with 2 elements"

    # collect metrics for each displacement pair in the following variable
    # initialize to -1 since none of the metrics considered can take negative values
    metrics = np.ones(frame1.shape+(np.product(searchwindow),), dtype=float) * -1

    for i in range(frame1.shape[0]):
        for j in range(frame1.shape[1]):
            print('Computing:',i,j,sep=' ',end='\n')
            metrics[i,j,:] = getmetrics(frame1, frame2, i, j, blocksize, searchwindow, metric, metrics)

    # find optimal d0, d1 estimates from evaluated distances in search space
    metrics[metrics==-1] = 1e10
    optimal = np.argmin(metrics, axis=2)
    optimal_d0 = optimal // searchwindow[0] - searchwindow[0] // 2 # check this
    optimal_d1 = optimal %  searchwindow[0] - searchwindow[1] // 2 # check this
    optimaldisp = np.stack((optimal_d0, optimal_d1), axis=2)

    return optimaldisp

### Main
def main():
    # loads frames and runs full-search algo on them

    # hyper-params
    blocksize = np.array([16, 16]).reshape(2,1)
    searchwindow = np.array([48, 48]).reshape(2,1)
    metric = 'mad'
    
    # load frames
    frame1 = io.imread('other-data-gray/Walking/frame10.png')
    frame2 = io.imread('other-data-gray/Walking/frame11.png')

    # call algo
    dispest = fsalgo(frame1, frame2, blocksize, searchwindow, metric)

    scipy.io.savemat('myresult.mat', dict(result=dispest))

    # compare with ground truth
    print(dispest.shape)

if __name__ == "__main__":
    main()