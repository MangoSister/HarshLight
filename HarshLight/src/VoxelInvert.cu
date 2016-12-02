#pragma once
#include "VoxelInvert.h"

__global__
void KernelTestWriteSurface(uint32_t voxel_dim, cudaSurfaceObject_t surf_obj);

void LaunchKernelVoxelInvert(uint32_t voxel_dim, cudaSurfaceObject_t surf_obj)
{
    dim3 block_dim(8, 8, 8);
    dim3 grid_dim(voxel_dim / block_dim.x, voxel_dim / block_dim.y, voxel_dim / block_dim.z);

    KernelTestWriteSurface << <grid_dim, block_dim >> >(voxel_dim, surf_obj);
    HANDLE_KERNEL_ERROR_SYNC;
}

__global__
void KernelTestWriteSurface(uint32_t voxel_dim, cudaSurfaceObject_t surf_obj)
{
    uint32_t x = blockIdx.x*blockDim.x + threadIdx.x;
    uint32_t y = blockIdx.y*blockDim.y + threadIdx.y;
    uint32_t z = blockIdx.z*blockDim.z + threadIdx.z;

    if (x >= voxel_dim || y >= voxel_dim || z >= voxel_dim)
        return;

    uint32_t element;
    surf3Dread(&element, surf_obj, x * sizeof(uint32_t), y, z);
	uint8_t r = (element & 0xFF000000) >> 24U;
	r = 0xFF - r;
	uint8_t g = (element & 0x00FF0000) >> 16U;
	g = 0xFF - g;
	uint8_t b = (element & 0x0000FF00) >> 8U;
	b = 0xFF - b;
	uint8_t a = (element & 0x000000FF);
	element = 
		(r & 0x000000FF) << 24U |
		(g & 0x000000FF) << 16U |
		(b & 0x000000FF) << 8U |
		(a & 0x000000FF);
    //element.x = 0xFF - element.x;
    //element.y = 0xFF - element.y;
    //element.z = 0xFF - element.z;
    //element.w = element.w;
    surf3Dwrite(element, surf_obj, x * sizeof(uint32_t), y, z);
}