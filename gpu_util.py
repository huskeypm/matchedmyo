#!/miniconda3/bin/python3.7
"""Set of routines for manipulation of images on GPUs using PyOpenCL."""

from __future__ import absolute_import, print_function
import sys
import numpy as np
import pyopencl as cl
import gputools

import time

def gpu_tester():
    """Test implementation of GPU routines thus far.

    This is meant as a temporary routine to debug GPU utilities that will eventually be used in the
    GPU portion of the MatchedMyo code.
    """
    #basic_test()
    bool_test()

    print ("\nTESTS HAVE PASSED!!")

def bool_test():
    """Taking a shot in the dark and trying to see if flattening the array will let us index
    the array via boolean masking"""
    start = time.time()

    #total_pixels = 512 * 512 * 256
    total_pixels = 100

    a_np = np.random.rand(total_pixels).astype(np.float32)
    b_np = np.random.rand(total_pixels).astype(np.float32)
    # create boolean indexing array
    #c_np = np.greater(np.random.rand(1000), 0.5)
    #cpu_indexes = np.asarray([0,3,7], np.int8)

    # transfer to gpu
    a_g = gputools.OCLArray.from_array(a_np)
    b_g = gputools.OCLArray.from_array(b_np)    
    #c_g = gputools.OCLArray.from_array(c_np)
    #gpu_indexes = gputools.OCLArray.from_array(cpu_indexes)

    # try boolean indexing
    #a_g.setitem(c_g, b_g)
    #a_g.setitem(gpu_indexes, b_g)

    max_bool_comparison_gpu = cl.array.maximum(a_g, b_g)

    start_transfer = time.time()
    max_bool_comparison_gpu_indexes = gputools.OCLArray.from_array(np.nonzero(max_bool_comparison_gpu.get()))
    index_transfer_time = time.time() - start_transfer
    print ("Index Transfer Time:", index_transfer_time)

    a_g.setitem(max_bool_comparison_gpu_indexes, b_g)

    a_new = a_g.get()
    b_new = b_g.get()

    #print ("Old a:",a_np)
    #print ("Old b:",b_np)
    #print ("Old c:",c_np)
    #print ()
    #print ("New a:",a_new)
    #print ("New b:",b_new)
    print (np.sum(a_new - a_np))

    total_time = time.time() - start

    print ("Total Computational Time:", total_time)
    print ("Transfer time percentage of total:", index_transfer_time / total_time)


def basic_test():
    """Gives the most basic test of PyOpenCL shown on their website."""
    a_np = np.random.rand(50000).astype(np.float32)
    b_np = np.random.rand(50000).astype(np.float32)

    ctx = cl.create_some_context()
    queue = cl.CommandQueue(ctx)

    mf = cl.mem_flags
    a_g = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=a_np)
    b_g = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=b_np)

    prg = cl.Program(ctx, """
    __kernel void sum(
        __global const float *a_g, __global const float *b_g, __global float *res_g)
    {
      int gid = get_global_id(0);
      res_g[gid] = a_g[gid] + b_g[gid];
    }
    """).build()

    res_g = cl.Buffer(ctx, mf.WRITE_ONLY, a_np.nbytes)
    prg.sum(queue, a_np.shape, None, a_g, b_g, res_g)

    res_np = np.empty_like(a_np)
    cl.enqueue_copy(queue, res_np, res_g)

    # Check on CPU with Numpy:
    print(res_np - (a_np + b_np))
    print(np.linalg.norm(res_np - (a_np + b_np)))
    assert np.allclose(res_np, a_np + b_np)

if __name__ == "__main__":
    for i,arg in enumerate(sys.argv):
        if arg=="--test" or arg=="-t":
            gpu_tester()
            quit()

    raise RuntimeError("Arguments not understood.")

