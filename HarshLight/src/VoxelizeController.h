#pragma once

#include "Util.h"
#include "Camera.h"
#include "Material.h"
#include "Component.h"
#include <cuda_gl_interop.h>
#include <cuda_runtime.h>

class VoxelizeController : public Component
{
public: 
	static const uint32_t s_VoxelChannelNum = 2; // s_VoxelFragmentSize * 32 bit / 4 bytes
	static const char* s_VoxelChannelNames[s_VoxelChannelNum];

    explicit VoxelizeController(uint32_t dim, float extent, Camera* voxel_cam);
	virtual ~VoxelizeController();

    void Start() override;
    void Update(float dt) override { }
    
    void SetVoxelDim(uint32_t dim);
    uint32_t GetVoxelDim() const;

	inline const Texture3dCompute* GetVoxelizeTex(uint32_t channel)
	{ return m_VoxelizeTex[channel]; }

    void TransferVoxelDataToCuda(cudaSurfaceObject_t surf_objs[s_VoxelChannelNum]);
	void FinishVoxelDataFromCuda(cudaSurfaceObject_t surf_objs[s_VoxelChannelNum]);

private:
	
	void DispatchVoxelization();

    static const char* s_VoxelDimName;
    static const char* s_ViewMtxToDownName;
    static const char* s_ViewMtxToLeftName;
    static const char* s_ViewMtxToForwardName;

    uint32_t m_VoxelDim;
    float m_Extent;
    Camera* m_VoxelCam;

	Texture3dCompute* m_VoxelizeTex[s_VoxelChannelNum];

	cudaGraphicsResource* m_CudaResources[s_VoxelChannelNum];
};