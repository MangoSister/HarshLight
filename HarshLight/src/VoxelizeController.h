#pragma once

#include "Util.h"
#include "Camera.h"
#include "Material.h"
#include "Component.h"
#include "Light.h"
#include "ComputeShaderProgram.h"
#include <cuda_gl_interop.h>
#include <cuda_runtime.h>
#include "glm/glm.hpp"


namespace VoxelChannel
{
	enum
	{
		TexVoxelAlbedo = 0,
		TexVoxelNormal = 1,
		TexVoxelRadiance = 2,
		Count = 3,
	};
};

class VoxelizeController : public Component
{
public: 
	static const char* s_VoxelChannelNames[VoxelChannel::Count];

    explicit VoxelizeController(uint32_t voxel_dim, uint32_t light_injection_res, const glm::vec3& center, const glm::vec3& extent, Camera* voxel_cam);
	virtual ~VoxelizeController();

    void Start() override;
	void Update(float dt) override;
    
    void SetVoxelDim(uint32_t dim);
    uint32_t GetVoxelDim() const;

	inline const Texture3dCompute* GetVoxelizeTex(uint32_t channel)
	{ return m_VoxelizeTex[channel]; }

    void TransferVoxelDataToCuda(cudaSurfaceObject_t surf_objs[VoxelChannel::Count]);
	void FinishVoxelDataFromCuda(cudaSurfaceObject_t surf_objs[VoxelChannel::Count]);

    inline GLuint GetDepthMap() const
    { return m_DepthMap; }

private:
	
	void DispatchVoxelization();
	void DispatchLightInjection();

    void LightSpaceBBox(const DirLight& light, glm::vec3& bmin, glm::vec3& bmax) const;

    static const char* s_VoxelDimName;
    static const char* s_ViewMtxToDownName;
    static const char* s_ViewMtxToLeftName;
    static const char* s_ViewMtxToForwardName;

    uint32_t m_VoxelDim = 256;
    glm::vec3 m_Center;
    glm::vec3 m_Extent;
    Camera* m_VoxelCam;

	Texture3dCompute* m_VoxelizeTex[VoxelChannel::Count];
	cudaGraphicsResource* m_CudaResources[VoxelChannel::Count];

	uint32_t m_LightInjectionRes = 1024;
	GLuint m_LightViewUBuffer;
	GLuint m_DepthFBO;
	GLuint m_DepthMap;

    ComputeShaderProgram* m_LightInjectionShader;
    uint32_t m_LightInjectionGroupSize = 8;
};