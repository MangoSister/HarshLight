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

	void EnableShadowSampling();
	void DisableShadowSampling();

    inline GLuint GetDirectionalDepthMap(uint32_t idx) const
    { return m_DirectionalDepthMap[idx]; }

    inline GLuint GetCubeDepthMap(uint32_t idx) const
    { return m_CubeDepthMap[idx]; }

	inline const Texture3dCompute* GetAnisoRadianceMipmap(uint32_t idx) const
	{ return m_AnisoRadianceMipmap[idx]; }

private:
	
	void DispatchVoxelization();
	void DispatchDirLightInjection();
    void DispatchPointLightInjection();
	void MipmapRadiance();

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

	uint32_t m_DirLightInjectionRes = 1024;
    uint32_t m_PointLightInjectionRes = 256;

	GLuint m_DirLightViewUBuffer;
	GLuint m_DepthFBO;
	GLuint m_DirectionalDepthMap[LightManager::s_DirLightMaxNum];
    GLuint m_CubeDepthMap[LightManager::s_PointLightMaxNum];
    GLuint m_PointLightCaptureUBuffer;
    static inline uint32_t GetPointLightCaptureUBufferSize()
    { return 7 * sizeof(glm::mat4x4) + sizeof(glm::vec4) + sizeof(glm::vec2); }

    ComputeShaderProgram* m_DirLightInjectionShader;
	ComputeShaderProgram* m_PointLightInjectionShader;
    uint32_t m_LightInjectionGroupSize = 8;


    static const uint32_t s_AnisotropicMipmapCount = 6;
	ComputeShaderProgram* m_AnisotropicMipmapShaderLeaf;
	ComputeShaderProgram* m_AnisotropicMipmapShaderInteriorBox;
    ComputeShaderProgram* m_AnisotropicMipmapShaderInterior[s_AnisotropicMipmapCount];

	uint32_t m_AnisoMipmapGroupSize = 8;
    uint32_t m_BoxMipmapGroupSize = 8;
	Texture3dCompute* m_AnisoRadianceMipmap[s_AnisotropicMipmapCount];
};