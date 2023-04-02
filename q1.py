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

def extractblocks(A, b1, b2, x, y, m, n):
    # Inputs:
    # A  = input block of some size
    # b1 = block size along dim 0
    # b2 = block size along dim 1
    # x  = starting position of block along dim 0
    # y  = starting position of block along dim 1
    # m  = frame size along dim 0
    # n  = frame size along dim 1
    # 
    # Returns:
    # List of frame-sets ~ enables unequal block sizes on edges
    # All frames in a frame-set have the same size ~ enables parallelism
    # Block measure functions can be called optimally on each frame-set


def compute(metrics, i, j, ):


def getmetrics(frame1, frame2, i, j, blocksize, searchwindow, metrics):
    localmetrics = np.ones(np.product(searchwindow)) * -1

    for d1 in range(searchwindow[0]):
        for d2 in range(searchwindow[1]):
            idx = d1*searchwindow[0]+d2
            vals = metrics[i-d1, j-d2, :]
            d1_ = d1 - searchwindow[0]//2
            d2_ = d2 - searhcwindow[1]//2
            if np.all(vals == -1):
                localmetrics[idx] = compute(frame1, frame2, i, j, d1_, d2_, blocksize, searchwindow)
            else:
                localmetrics[idx] = val[idx]

    return localmetrics


### Algo
def fsalgo(frame1, frame2, blocksize, searchwindow):
    # overall computation of full-search and returns d1,d2 estimates 
    # uses subroutines to split tasks
    # uses memoization to avoid redoing computation

    assert frame.shape == frame2.shape, "Frames are not of the same shape"
    assert np.count_nonzero(blocksize.shape)==2, "Block size should be a tuple with 2 elements"

    # collect metrics for each displacement pair in the following variable
    # initialize to -1 since none of the metrics considered can take negative values
    metrics = np.ones(frame1.shape+(np.product(blocksize),), dtype=float) * -1

    for i in range(frame1.shape[0]):
        for j in range(frame1.shape[1]):
            metrics[i,j,:] = getmetrics(frame1, frame2, i, j, blocksize, searchwindow, metrics)

    optimal = np.argmin(metrics, axis=2)
    optimal_d1 = optimal // searchwindow[0] - searchwindow[0] // 2 # check this
    optimal_d2 = optimal %  searchwindow[1] - searchwindow[1] // 2 # check this
    optimaldisp = np.stack(optimal_d1, optimal_d2, axis=2)

    return optimaldisp

### Main
def main():
    # loads frames and runs full-search algo on them

    # hyper-params
    blocksize = [16, 16]
    searchwindow = [48, 48]
    
    # load frames
    
    # call algo
    dispest = fsalgo(frame1, frame2, blocksize, searchwindow)

    # compare with ground truth
