#pragma once
#include <cuda_runtime.h>
#include <device_launch_parameters.h>

#include "Util.h"

void LaunchKernelVoxelInvert(uint32_t voxel_dim, cudaSurfaceObject_t surf_obj);
