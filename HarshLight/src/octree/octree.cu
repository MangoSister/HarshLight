#include <cuda.h>
#include <cuda_runtime.h>
#include <driver_functions.h>

#include <cstdio>

//const int block_size = 16;
/*
   0-5 color +-x, +-y, +-z;
   6 normals
   7 father
   8 child
   9 info
   last byte
1: is procesing
2: is leaves
4: is first son

10 light
 */
const int block_size = 11;
const int idx_normal = 6;
const int idx_father = 7;
const int idx_child = 8;
const int idx_info = 9;
const int memory_size = 2000000;

const int leave_size = 3;

/*
leaves:
r g b a
x y z a
c: x y z

queue:
leavesidx, treeidx, level,
 */

__device__ int
cudaCombine(int a, int b, int c, int d) {
    return ((a & 0xFF) << 24) + ((b & 0xFF) << 16)
        + ((c & 0xFF) << 8) + (d & 0xFF);
}

/*****************************************************************************/
/**Find all valid leaves******************************************************/
/*****************************************************************************/
__global__ void
cudaFindLeavesThreads(int* colors, int* normals, int* leaves, int* p, int dim) {
    int x = threadIdx.x;
    int y = blockIdx.x % blockDim.x;
    int z = blockIdx.x / blockDim.x;

    int element;
    element = colors[x * dim * dim + y * dim + z];
    //     	surf3Dread(&element, colors, x * sizeof(uint32_t), y, z);
    if (element & 0xFF) {
        int idx = atomicAdd(p, 1) * leave_size;
        leaves[idx] = element;
        leaves[idx + 1] = normals[x * dim * dim + y * dim + z];
        leaves[idx + 2] = cudaCombine(x, y, z, 0);
    }
}

int cudaFindLeaves(int* colors, int* normals, int* leaves, int* p, int dim) {
    cudaMemset(p, 0, sizeof(int));
    int threadsPerBlock = dim;
    int blocks = dim * dim;
    cudaFindLeavesThreads<<<blocks, threadsPerBlock>>>(colors, normals, leaves, p, dim);

    int num_leaves;
    cudaMemcpy(&num_leaves, p, sizeof(int), cudaMemcpyDeviceToHost);
    return num_leaves;
}

/*****************************************************************************/
/**build tree details*********************************************************/
/*****************************************************************************/
__device__ void
cudaSetLeave(int tree_idx, int* tree, int* leave) {
    for (int i = 0; i < 6; ++i) {
        tree[i] = leave[0];
    }
    tree[idx_normal] = leave[1];
    leave[2] = tree_idx;
}

__device__ void
cudaAddToTree(int leave_idx, int* leaves,
        int tree_idx, int* tree, int level, int* ptrTree,
        int* queue, int* ptrQueue) {

    int* tree_element = &tree[tree_idx * block_size];

    // is leaves
    if (level == 1) {
        cudaSetLeave(tree_idx, tree_element, &leaves[leave_idx * leave_size]);
        tree_element[idx_info] += 2;
        return ;
    }
    // has child finish processing
    if (tree_element[idx_child] && (tree_element[idx_info] & 1) == 0) {		
        int* leave_element = &leaves[leave_idx * leave_size];
        int new_level = level >> 1;
        bool x = ((leave_element[2] & 0xFF000000) >> 24) & new_level;
        bool y = ((leave_element[2] & 0x00FF0000) >> 16) & new_level;
        bool z = ((leave_element[2] & 0x0000FF00) >>  8) & new_level;
        int new_idx = tree_element[idx_child] + x * 4 + y * 2 + z;
        cudaAddToTree(leave_idx, leaves, new_idx, tree, new_level, ptrTree, queue, ptrQueue);
    } else {
        // try to process
        bool processing = atomicOr(&tree_element[idx_info], 1) & 1;
        if (processing) {
            int ptr = atomicAdd(ptrQueue, 1);
            queue[ptr * 3] = leave_idx;
            queue[ptr * 3 + 1] = tree_idx;
            queue[ptr * 3 + 2] = level;
        } else {
            // process
            int idx = atomicAdd(ptrTree, 8) + 1;
            tree_element[idx_child] = idx;
            for (int i = 0; i < 8; ++i) {
                tree[(idx + i) * block_size + idx_father] = tree_idx;
            }
            tree[idx * block_size + idx_info] += 4;

            // finish processing
            atomicAnd(&tree_element[idx_info], 0xFFFFFFFE);

            int* leave_element = &leaves[leave_idx * leave_size];
            int new_level = level >> 1;
            bool x = ((leave_element[2] & 0xFF000000) >> 24) & new_level;
            bool y = ((leave_element[2] & 0x00FF0000) >> 16) & new_level;
            bool z = ((leave_element[2] & 0x0000FF00) >>  8) & new_level;
            int new_idx = tree_element[idx_child] + x * 4 + y * 2 + z;
            cudaAddToTree(leave_idx, leaves, new_idx, tree, new_level, ptrTree, queue, ptrQueue);
        }
    }
}

__global__ void
cudaBuildTreeFromLeaves(int* leaves, int num_leaves, int* tree, int* ptrTree, int* queue, int* ptrQueue, int dim) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < num_leaves) {
        cudaAddToTree(idx, leaves, 0, tree, dim, ptrTree, queue, ptrQueue);
    }
}

__global__ void
cudaBuildTreeFromQueue(int* leaves, int num_leaves, int* tree, int* ptrTree, int* queue, int* ptrQueue, int rangeL, int rangeR, int dim) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x + rangeL;
    if (idx < rangeR) {
        int* queue_element = &queue[idx * leave_size];
        cudaAddToTree(queue_element[0], leaves, queue_element[1], tree, queue_element[2], ptrTree, queue, ptrQueue);
    }
}

void cudaBuildTree(int* leaves, int num_leaves, int* queue, int* tree, int dim, int* ptrTree, int* ptrQueue) {
    cudaMemset(ptrTree, 0, sizeof(int));
    cudaMemset(ptrQueue, 0, sizeof(int));

    int fr, la, total = dim * dim * dim;
    int threadsPerBlock = dim, blocks = (num_leaves - 1 + threadsPerBlock) / threadsPerBlock;
    cudaBuildTreeFromLeaves<<<blocks, threadsPerBlock>>>(leaves, num_leaves, tree, ptrTree, queue, ptrQueue, dim);

    fr = 0;
    cudaMemcpy(&la, ptrQueue, sizeof(int), cudaMemcpyDeviceToHost);
    while (la > fr) {
        if (la >= total) {
            cudaMemset(ptrQueue, 0, sizeof(int));
        }

        blocks = (la - fr - 1 + threadsPerBlock) / threadsPerBlock;
        cudaBuildTreeFromQueue<<<blocks, threadsPerBlock>>>(leaves, num_leaves, tree, ptrTree, queue, ptrQueue, fr, la, dim);

        fr = la >= total ? 0:la;
        cudaMemcpy(&la, ptrQueue, sizeof(int), cudaMemcpyDeviceToHost);
    }
}

/*****************************************************************************/
/***mix up info***************************************************************/
/*****************************************************************************/
__global__ void
cudaCombineFromLeavesThread(int* tree, int* leaves, int num_leaves, int* queue, int* ptrQueue) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < num_leaves) {
        int tree_idx = leaves[idx * leave_size + 2];
        int* element = &tree[tree_idx * block_size];
        if (element[idx_info] & 4) {
            int new_idx = atomicAdd(ptrQueue, 1);
            queue[new_idx] = element[idx_father];
        }
    }
}

__device__ void
addColor(int* c0, int* c1) {
    c0[0] += ((*c1) & 0xFF000000) >> 24;
    c0[1] += ((*c1) & 0x00FF0000) >> 16;
    c0[2] += ((*c1) & 0x0000FF00) >> 8; 
    c0[3] += (*c1) & 0xFF;
}

__device__ int
alphaBlend(int* c0, int* c1) {
    int a0 = c0[3], a1 = c1[3], a_new = 255 - a0;

    int r = int((c0[0] * a0 + c1[0] * a_new) / 4 / 255.0f) & 0xFF;
    int g = int((c0[1] * a0 + c1[1] * a_new) / 4 / 255.0f) & 0xFF;
    int b = int((c0[2] * a0 + c1[2] * a_new) / 4 / 255.0f) & 0xFF;
    int a = int((c0[0] * a0 + a1 * a_new) / 4) & 0xFF;

    return cudaCombine(r, g, b, a);
}

__global__ void
cudaCombineFromQueueThread(int* tree, int rangeL, int rangeR, int* queue, int* ptrQueue) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x + rangeL;
    if (idx < rangeR) {
        int* element = &tree[queue[idx] * block_size];

        // color
        int xx[16], yy[16], zz[16];
        memset(xx, 0, sizeof(xx));
        memset(yy, 0, sizeof(yy));
        memset(zz, 0, sizeof(zz));

        for (int x = 0; x < 2; ++x) {
            for (int y = 0; y < 2; ++y) {
                for (int z = 0; z < 2; ++z) {
                    int idx_i = x * 4 + y * 2 + z;

                    if (x) {
                        addColor(&xx[0], &tree[(element[idx_child] + idx_i) * block_size]);
                        addColor(&xx[12], &tree[(element[idx_child] + idx_i) * block_size + 1]);
                    } else {
                        addColor(&xx[4], &tree[(element[idx_child] + idx_i) * block_size]);
                        addColor(&xx[8], &tree[(element[idx_child] + idx_i) * block_size + 1]);
                    }

                    if (y) {
                        addColor(&yy[0], &tree[(element[idx_child] + idx_i) * block_size + 2]);
                        addColor(&yy[12], &tree[(element[idx_child] + idx_i) * block_size + 3]);
                    } else {
                        addColor(&yy[4], &tree[(element[idx_child] + idx_i) * block_size + 2]);
                        addColor(&yy[8], &tree[(element[idx_child] + idx_i) * block_size + 3]);
                    }

                    if (z) {
                        addColor(&zz[0], &tree[(element[idx_child] + idx_i) * block_size + 4]);
                        addColor(&zz[12], &tree[(element[idx_child] + idx_i) * block_size + 5]);
                    } else {
                        addColor(&zz[4], &tree[(element[idx_child] + idx_i) * block_size + 4]);
                        addColor(&zz[8], &tree[(element[idx_child] + idx_i) * block_size + 5]);
                    }
                }
            }
        }

        element[0] = alphaBlend(&xx[0], &xx[4]);
        element[1] = alphaBlend(&xx[8], &xx[12]);
        element[2] = alphaBlend(&yy[0], &yy[4]);
        element[3] = alphaBlend(&yy[8], &yy[12]);
        element[4] = alphaBlend(&zz[0], &zz[4]);
        element[5] = alphaBlend(&zz[8], &zz[12]);

        //		element[6] = normalMix();

        if (element[idx_info] & 4) {
            int new_idx = atomicAdd(ptrQueue, 1);
            queue[new_idx] = element[idx_father];
        }
    }
}

void cudaCombine(int* tree, int* leaves, int num_leaves, int* queue, int* ptrQueue) {
    int threadsPerBlock = 256;
    int blocks = (num_leaves - 1 + threadsPerBlock) / threadsPerBlock;
    cudaMemset(ptrQueue, 0, sizeof(int));

    int st = 0, en;
    cudaCombineFromLeavesThread<<<blocks, threadsPerBlock>>>(tree, leaves, num_leaves, queue, ptrQueue);

    cudaMemcpy(&en, ptrQueue, sizeof(int), cudaMemcpyDeviceToHost);
    while (en > st) {
        blocks = (en - st - 1 + threadsPerBlock) / threadsPerBlock;
        cudaCombineFromQueueThread<<<blocks, threadsPerBlock>>>(tree, st, en, queue, ptrQueue);

        st = en;
        cudaMemcpy(&en, ptrQueue, sizeof(int), cudaMemcpyDeviceToHost);
    }
}

/*****************************************************************************/
/***wrap up*******************************************************************/
/*****************************************************************************/
int cudaBuildTreeOverall(int* colors, int* normals, int* leaves, int* tree, int* queue, int dim) {
    int *ptr0, *ptr1;
    cudaMalloc(&ptr0, sizeof(int));
    cudaMalloc(&ptr1, sizeof(int));
    cudaMemset(tree, 0, sizeof(int) * memory_size * block_size);

    int num_leaves = cudaFindLeaves(colors, normals, leaves, ptr0, dim);
    cudaBuildTree(leaves, num_leaves, queue, tree, dim, ptr0, ptr1);

    cudaCombine(tree, leaves, num_leaves, queue, ptr0);

    cudaFree(ptr0);
    cudaFree(ptr1);

    return num_leaves;
}

// build tree for test
void cudaBuildMalloc(int* colors, int* normals, int voxel_dim, int* res) {
    int *gpu_colors, *gpu_normals, *gpu_tmp, *gpu_res, *gpu_queue;

    int total = voxel_dim * voxel_dim * voxel_dim;
    cudaMalloc(&gpu_colors, sizeof(int) * total);
    cudaMemcpy(gpu_colors, colors, sizeof(int) * total, cudaMemcpyHostToDevice);
    cudaMalloc(&gpu_normals, sizeof(int) * total);
    cudaMemcpy(gpu_normals, normals, sizeof(int) * total, cudaMemcpyHostToDevice);
    cudaMalloc(&gpu_tmp, sizeof(int) * memory_size * 6);
    cudaMalloc(&gpu_res, sizeof(int) * memory_size * block_size);
    cudaMalloc(&gpu_queue, sizeof(int) * total);

    int len = cudaBuildTreeOverall(gpu_colors, gpu_normals, gpu_tmp, gpu_res, gpu_queue, voxel_dim);
    printf("len = %d\n", len);
}

/*****************************************************************************/

void printCudaInfo()
{
    // for fun, just print out some stats on the machine

    int deviceCount = 0;
    cudaError_t err = cudaGetDeviceCount(&deviceCount);

    printf("---------------------------------------------------------\n");
    printf("Found %d CUDA devices\n", deviceCount);

    for (int i=0; i<deviceCount; i++)
    {
        cudaDeviceProp deviceProps;
        cudaGetDeviceProperties(&deviceProps, i);
        printf("Device %d: %s\n", i, deviceProps.name);
        printf("   SMs:        %d\n", deviceProps.multiProcessorCount);
        printf("   Global mem: %.0f MB\n",
                static_cast<float>(deviceProps.totalGlobalMem) / (1024 * 1024));
        printf("   CUDA Cap:   %d.%d\n", deviceProps.major, deviceProps.minor);
    }
    printf("---------------------------------------------------------\n"); 
}
