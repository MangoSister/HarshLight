#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/type_ptr.hpp>

#include "VoxelizeController.h"
#include "ModelRenderer.h"
#include "Actor.h"
#include "World.h"
#include "Camera.h"


const char* VoxelizeController::s_VoxelDimName = "VoxelDim";
const char* VoxelizeController::s_ViewMtxToDownName = "ViewMtxToDown";
const char* VoxelizeController::s_ViewMtxToLeftName = "ViewMtxToLeft";
const char* VoxelizeController::s_ViewMtxToForwardName = "ViewMtxToForward";

const char* VoxelizeController::s_VoxelChannelNames[VoxelChannel::Count]
{
	"TexVoxelAlbedo",
	"TexVoxelNormal",
	"TexVoxelRadiance",
};

VoxelizeController::VoxelizeController(uint32_t voxel_dim, uint32_t light_injection_res, const glm::vec3& center, const glm::vec3& extent, Camera* voxel_cam)
    :Component(), m_VoxelDim(voxel_dim), m_Center(center), m_Extent(extent), m_VoxelCam(voxel_cam), m_LightInjectionRes(light_injection_res)
{
	for (uint32_t i = 0; i < VoxelChannel::Count; i++)
		m_VoxelizeTex[i] = new Texture3dCompute(voxel_dim, voxel_dim, voxel_dim, GL_R32UI, GL_RED_INTEGER, GL_UNSIGNED_INT);

	glGenBuffers(1, &m_LightViewUBuffer);
	glBindBuffer(GL_UNIFORM_BUFFER, m_LightViewUBuffer);
	glBufferData(GL_UNIFORM_BUFFER, Camera::GetUBufferSize(), nullptr, GL_STATIC_DRAW);
	glBindBuffer(GL_UNIFORM_BUFFER, 0);

	glGenFramebuffers(1, &m_DepthFBO);

	glGenTextures(1, &m_DepthMap);
	glBindTexture(GL_TEXTURE_2D, m_DepthMap);
	glTexImage2D(GL_TEXTURE_2D, 0, GL_DEPTH_COMPONENT,
		m_LightInjectionRes, m_LightInjectionRes, 0, GL_DEPTH_COMPONENT, GL_FLOAT, NULL);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_REPEAT);

	glBindFramebuffer(GL_FRAMEBUFFER, m_DepthFBO);
	glFramebufferTexture2D(GL_FRAMEBUFFER, GL_DEPTH_ATTACHMENT, GL_TEXTURE_2D, m_DepthMap, 0);
	glDrawBuffer(GL_NONE);
	glReadBuffer(GL_NONE);
	glBindFramebuffer(GL_FRAMEBUFFER, 0);
}

VoxelizeController::~VoxelizeController()
{
	//unregister voxel_tex from cuda
	for (uint32_t i = 0; i < VoxelChannel::Count; i++)
	{
		cudaCheckError(cudaGraphicsUnregisterResource(m_CudaResources[i]));
		if (m_VoxelizeTex[i])
		{
			delete m_VoxelizeTex[i];
			m_VoxelizeTex[i] = nullptr;
		}
	}

	if (m_LightViewUBuffer)
	{
		glDeleteBuffers(1, &m_LightViewUBuffer);
		m_LightViewUBuffer = 0;
	}

	if (m_DepthMap)
	{
		glDeleteTextures(1, &m_DepthMap);
		m_DepthMap = 0;
	}

	if (m_DepthFBO)
	{
		glDeleteFramebuffers(1, &m_DepthFBO);
		m_DepthFBO = 0;
	}
}


void VoxelizeController::Start()
{
    glm::mat4x4 old_transform = m_VoxelCam->GetTransform();

    const float max_extent = std::max(m_Extent.x, std::max(m_Extent.y, m_Extent.z));

    glm::vec3 pos = m_Center + glm::vec3(0.0f, max_extent, 0.0f);
    m_VoxelCam->MoveTo(pos);
    m_VoxelCam->LookAtDir(glm::vec3(0.0f, -1.0f, 0.0f), glm::vec3(0.0f, 0.0f, 1.0f));
    glm::mat4x4 view_to_down = m_VoxelCam->GetViewMtx();

    pos = m_Center + glm::vec3(max_extent, 0.0, 0.0f);
    m_VoxelCam->MoveTo(pos);
    m_VoxelCam->LookAtDir(glm::vec3(-1.0f, 0.0f, 0.0f), glm::vec3(0.0f, 1.0f, 0.0f));
    glm::mat4x4 view_to_left = m_VoxelCam->GetViewMtx();

    pos = m_Center + glm::vec3(0.0f, 0.0f, -max_extent);
    m_VoxelCam->MoveTo(pos);
    m_VoxelCam->LookAtDir(glm::vec3(0.0f, 0.0f, 1.0f), glm::vec3(0.0f, 1.0f, 0.0f));
    glm::mat4x4 view_to_forward = m_VoxelCam->GetViewMtx();

    m_VoxelCam->SetTransform(old_transform);

	const std::vector<Material*>& voxel_mats = m_Actor->GetRenderer<ModelRenderer>()->GetMaterial(RenderPass::kVoxelize);
	for (auto mat : voxel_mats)
	{
        if (mat)
        {
            mat->SetVec2Param(s_VoxelDimName, glm::vec2(static_cast<float>(m_VoxelDim), static_cast<float>(m_VoxelDim)));
            mat->SetMat4x4Param(s_ViewMtxToDownName, view_to_down);
            mat->SetMat4x4Param(s_ViewMtxToLeftName, view_to_left);
            mat->SetMat4x4Param(s_ViewMtxToForwardName, view_to_forward);
        }
	}

	const std::vector<Material*>& regular_mats = m_Actor->GetRenderer<ModelRenderer>()->GetMaterial(RenderPass::kRegular);
	for (auto mat : regular_mats)
	{
        if (mat)
        {
            mat->SetVec2Param(s_VoxelDimName, glm::vec2(static_cast<float>(m_VoxelDim), static_cast<float>(m_VoxelDim)));
            mat->SetMat4x4Param(s_ViewMtxToDownName, view_to_down);
            mat->SetMat4x4Param(s_ViewMtxToLeftName, view_to_left);
            mat->SetMat4x4Param(s_ViewMtxToForwardName, view_to_forward);
        }
	}

	//register voxel texture to cuda
	for (uint32_t i = 0; i < VoxelChannel::Count; i++)
	{
		cudaCheckError(cudaGraphicsGLRegisterImage(&m_CudaResources[i], m_VoxelizeTex[i]->GetTexObj(),
			GL_TEXTURE_3D, cudaGraphicsRegisterFlagsSurfaceLoadStore));
	}
	
	DispatchVoxelization();
}

void VoxelizeController::Update(float dt)
{
	DispatchLightInjection();
}

void VoxelizeController::SetVoxelDim(uint32_t dim)
{
    m_VoxelDim = dim;
}

uint32_t VoxelizeController::GetVoxelDim() const
{
    return m_VoxelDim;
}

void VoxelizeController::TransferVoxelDataToCuda(cudaSurfaceObject_t surf_objs[VoxelChannel::Count])
{
	cudaCheckError(cudaGraphicsMapResources(VoxelChannel::Count, m_CudaResources, 0));
	
	for (uint32_t i = 0; i < VoxelChannel::Count; i++)
	{		
		cudaArray* channel_cuda_array;
		cudaCheckError(cudaGraphicsSubResourceGetMappedArray(&channel_cuda_array, m_CudaResources[i], 0, 0));

		struct cudaResourceDesc res_desc;
		memset(&res_desc, 0, sizeof(cudaResourceDesc));
		res_desc.resType = cudaResourceTypeArray;    // be sure to set the resource type to cudaResourceTypeArray
		res_desc.res.array.array = channel_cuda_array;

		cudaCheckError(cudaCreateSurfaceObject(&surf_objs[i], &res_desc));
	}
}

void VoxelizeController::FinishVoxelDataFromCuda(cudaSurfaceObject_t surf_objs[VoxelChannel::Count])
{
    //there is no unbinding surface API
	for (uint32_t i = 0; i < VoxelChannel::Count; i++)
		cudaCheckError(cudaDestroySurfaceObject(surf_objs[i]));

	cudaCheckError(cudaGraphicsUnmapResources(VoxelChannel::Count, m_CudaResources, 0));
}

void VoxelizeController::DispatchVoxelization()
{
	for (uint32_t i = 0; i < VoxelChannel::Count; i++)
		if (m_VoxelizeTex[i] == nullptr) return;
	for (uint32_t i = 0; i < VoxelChannel::Count; i++)
		m_VoxelizeTex[i]->CleanContent();

	glDisable(GL_DEPTH_TEST);
	glDisable(GL_CULL_FACE);
	glPolygonMode(GL_FRONT_AND_BACK, GL_FILL);
	glColorMask(GL_FALSE, GL_FALSE, GL_FALSE, GL_FALSE);
	glDepthMask(GL_FALSE);
	glViewport(0, 0, m_VoxelDim, m_VoxelDim);

	if (m_VoxelCam)
		m_VoxelCam->UpdateCamMtx(UniformBufferBinding::kMainCam);
	else
		fprintf(stderr, "WARNING: VoxelizeCamera is null\n");

	const RendererList& renderers = World::GetInst().GetRenderers();
	for (ModelRenderer* renderer : renderers)
	{
		renderer->Render(RenderPass::kVoxelize);		
	}

	glMemoryBarrier(GL_ALL_BARRIER_BITS);

	glEnable(GL_DEPTH_TEST);
	glEnable(GL_CULL_FACE);
	glColorMask(GL_TRUE, GL_TRUE, GL_TRUE, GL_TRUE);
	glDepthMask(GL_TRUE);
	uint32_t vw, vh;
	World::GetInst().GetViewportSize(vw, vh);
	glViewport(0, 0, vw, vh);

}

void VoxelizeController::DispatchLightInjection()
{
	for (uint32_t i = 0; i < VoxelChannel::Count; i++)
		if (m_VoxelizeTex[i] == nullptr) return;

	uint32_t vw, vh;
	World::GetInst().GetViewportSize(vw, vh);

    glBindFramebuffer(GL_FRAMEBUFFER, m_DepthFBO);
	glEnable(GL_DEPTH_TEST);
	glEnable(GL_CULL_FACE);
	glPolygonMode(GL_FRONT_AND_BACK, GL_FILL);
	glColorMask(GL_TRUE, GL_TRUE, GL_TRUE, GL_TRUE);
	glDepthMask(GL_TRUE);
	glViewport(0, 0, m_LightInjectionRes, m_LightInjectionRes);

	const LightManager& light_manager = World::GetInst().GetLightManager();
	const RendererList& renderers = World::GetInst().GetRenderers();

	for (uint32_t i = 0; i < light_manager.GetDirLightCount(); i++)
	{
		const DirLight& dir_light = light_manager.GetDirLight(i);
		//overwrite main camera

		glBindBufferRange(GL_UNIFORM_BUFFER, (uint8_t)UniformBufferBinding::kMainCam, m_LightViewUBuffer, 0, Camera::GetUBufferSize());
		glBindBuffer(GL_UNIFORM_BUFFER, m_LightViewUBuffer);
		const glm::mat4x4& light_mtx = dir_light.m_LightMtx;
        //compute light space bounding box
        glm::vec3 min, max;
        LightSpaceBBox(dir_light, min, max);
        glm::mat4x4 light_proj_mtx = glm::ortho(min.x, max.x, min.y, max.y, -max.z, -min.z);

        glBufferSubData(GL_UNIFORM_BUFFER, 0, sizeof(glm::mat4), glm::value_ptr(light_mtx));
        glBufferSubData(GL_UNIFORM_BUFFER, sizeof(glm::mat4), sizeof(glm::mat4), glm::value_ptr(light_proj_mtx));
        glBufferSubData(GL_UNIFORM_BUFFER, 2 * sizeof(glm::mat4), sizeof(glm::vec4), glm::value_ptr(glm::vec4(0.0f, 0.0f, 0.0f, 1.0f)));
		glBindBuffer(GL_UNIFORM_BUFFER, 0);

        glClear(GL_DEPTH_BUFFER_BIT);
		for (ModelRenderer* renderer : renderers)
			renderer->Render(RenderPass::kLightInjection);

        //now we have depth for current light, inject current light info

        glMemoryBarrier(GL_ALL_BARRIER_BITS);

	}
	
    glBindFramebuffer(GL_FRAMEBUFFER, 0);
	glEnable(GL_DEPTH_TEST);
	glEnable(GL_CULL_FACE);
	glColorMask(GL_TRUE, GL_TRUE, GL_TRUE, GL_TRUE);
	glDepthMask(GL_TRUE);
	glViewport(0, 0, vw, vh);
}

void VoxelizeController::LightSpaceBBox(const DirLight& light, glm::vec3& min, glm::vec3& max) const
{
    min.x = min.y = min.z = FLT_MAX;
    max.x = max.y = max.z = FLT_MIN;

    glm::vec4 center4(m_Center.x, m_Center.y, m_Center.z, 0.0f);

    glm::vec4 bbox[8]
    {
        center4 + glm::vec4{ -m_Extent.x, -m_Extent.y, -m_Extent.z, 1.0f },
        center4 + glm::vec4{ -m_Extent.x, -m_Extent.y, m_Extent.z, 1.0f },
        center4 + glm::vec4{ -m_Extent.x, m_Extent.y, -m_Extent.z, 1.0f },
        center4 + glm::vec4{ -m_Extent.x, m_Extent.y, m_Extent.z, 1.0f },

        center4 + glm::vec4{ m_Extent.x, -m_Extent.y, -m_Extent.z, 1.0f },
        center4 + glm::vec4{ m_Extent.x, -m_Extent.y, m_Extent.z, 1.0f },
        center4 + glm::vec4{ m_Extent.x, m_Extent.y, -m_Extent.z, 1.0f },
        center4 + glm::vec4{ m_Extent.x, m_Extent.y, m_Extent.z, 1.0f },
    };

    for (uint32_t i = 0; i < 8; i++)
    {
        glm::vec4 curr = light.m_LightMtx * bbox[i];
        if (curr.x < min.x)
            min.x = curr.x;
        if (curr.x > max.x)
            max.x = curr.x;

        if (curr.y < min.y)
            min.y = curr.y;
        if (curr.y > max.y)
            max.y = curr.y;

        if (curr.z < min.z)
            min.z = curr.z;
        if (curr.z > max.z)
            max.z = curr.z;
    }
}
