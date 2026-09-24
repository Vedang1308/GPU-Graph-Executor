In this assignment, we would implement a GPU graph executor that can train simple neural nets such as multilayer perceptron models.

Our code (assignment2.tar.gz) should be able to construct a simple MLP model using computation graph API implemented in Assignment 2, and train and test the model using either numpy or GPU. If you implement everything correctly, you would see nice speedup in training neural nets with GPU executor compared to numpy executor, as expected. 

Key concepts and data structures that we would need to implement are:

 - Shape inference on computation graph given input shapes.
 - GPU executor memory management for computation graph.
 - GPU kernel implementations of common kernels, e.g. Relu, MatMul, Softmax.


Overview of Module

Code to download: assignment2.tar.gz

 - python/dlsys/autodiff.py: Implements computation graph, autodiff, GPU/Numpy Executor.

 - python/dlsys/gpu_op.py: Exposes Python function to call GPU kernels via ctypes.

 - python/dlsys/ndarray.py: Exposes Python GPU array API.

 - src/dlarray.h: header for GPU array.

 - src/c_runtime_api.h: C API header for GPU array and GPU kernels.

 - src/gpu_op.cu: cuda implementation of kernels

Overview of Task
Understand the code skeleton and tests. Fill in implementation wherever marked """TODO: Your code here""".

There are only two files with TODOs for you. Please finish these two files and submit these two files to gradescope.

 - python/dlsys/autodiff.py
 - src/gpu_op.cu


**Special note**
Do not change Makefile to use cuDNN for GPU kernels.

Environment setup
 - If you don't have a GPU machine, you can use AWS GPU instance. 
 - AWS GPU instances require a fee. You can also try free resources such as Google Colab, or ASU Research Computing
 - Otherwise, you need to install CUDA toolkit (https://docs.nvidia.com/cuda/cuda-installation-guide-linux/) on your own machine, and set the environment variables.

export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH

Tests cases
We have 12 tests in tests/test_gpu_op.py. We would grade your GPU kernel implementations based on those tests. We would also grade your implementation of shape inference and memory management based on tests/mnist_dlsys.py.


Compile

export PYTHONPATH="${PYTHONPATH}:/path/to/assignment4/python"
make

Note: before MAKE, please check your GPU and CUDA is compatible with the CUDA gencode and CUDA arch. You can check this link (https://arnon.dk/tag/cuda-arch/) for help. You may need to modify the ARCH in Makefile correspondingly.
For example, CUDA 11 does not support arch=compute_30,code=sm_30. If you are running CUDA 11, you need to remove this before MAKE.


Run all tests with

# sudo pip install nose
nosetests -v tests/test_gpu_op.py


Run neural nets training

# see cmd options with
#get help
python tests/mnist_dlsys.py -h

# run logistic regression on numpy
python tests/mnist_dlsys.py -l -m logreg -c numpy

# run logistic regression on gpu
python tests/mnist_dlsys.py -l -m logreg -c gpu

# run MLP on numpy
python tests/mnist_dlsys.py -l -m mlp -c numpy

# run MLP on gpu
python tests/mnist_dlsys.py -l -m mlp -c gpu

If your implementation is correct, you would see
 - generally decreasing loss value with epochs, similar loss value decrease for numpy and GPU execution
 - your dev set accuracy for logreg about 92% and MLP about 97% for mnist using the parameters we provided in mnist_dlsys.py
 - GPU execution being noticeably faster than numpy.
However, if you do not reuse memory across executor.runs, your GPU execution would incur overhead in memory allocation.

Profile GPU execution with:
nvprof python tests/mnist_dlsys.py -l -m mlp -c gpu

Note: If your GPU's compute capability is greater than 7.2, you need to use Nsight Compute to profile your code by running the following code:
nsys nvprof python tests/mnist_dlsys.py -l -m mlp -c gpu

If GPU memory management is done right, e.g. reuse GPU memory across each executor.run, your cudaMalloc "Calls" should not increase with number of training epochs (set with -e option).

# Run 10 epochs nvprof python tests/mnist_dlsys.py -l -m mlp -c gpu -e 10

#==2263== API calls:

#Time(%) Time Calls Avg Min Max Name

# 10.19% 218.65ms 64 3.4164ms 8.5130us 213.90ms cudaMalloc

# Run 30 epochs nvprof python tests/mnist_dlsys.py -l -m mlp -c gpu -e 30

#==4333== API calls: #Time(%) Time Calls Avg Min Max Name

# 5.80% 340.74ms 64 5.3240ms 15.877us 333.80ms cudaMalloc



**Grading rubrics**

This assignment will map to 20 points. 

 - test_gpu_op.test_array_set ... 1 pt
 - test_gpu_op.test_broadcast_to ... 1 pt
 - test_gpu_op.test_reduce_sum_axis_zero ... 1 pt
 - test_gpu_op.test_matrix_elementwise_add ... 1 pt
 - test_gpu_op.test_matrix_elementwise_add_by_const ... 1 pt
 - test_gpu_op.test_matrix_elementwise_multiply ... 1 pt
 - test_gpu_op.test_matrix_elementwise_multiply_by_const ... 1 pt
 - test_gpu_op.test_matrix_multiply ... 2 pt
 - test_gpu_op.test_relu ... 2 pt
 - test_gpu_op.test_relu_gradient ... 2 pt
 - test_gpu_op.test_softmax ... 2 pt
 - test_gpu_op.test_softmax_cross_entropy ... Implemented
 - mnist with MLP using numpy ... 2 pt
 - mnist with MLP using gpu ... 3 pt


Submitting your work
Please submit your code to GradeScope. You only need submit two files:
 - python/dlsys/autodiff.py
 - src/gpu_op.cu