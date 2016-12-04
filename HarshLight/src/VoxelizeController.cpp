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

VoxelizeController::VoxelizeController(uint32_t voxel_dim, uint32_t light_injection_res, float extent, Camera* voxel_cam)
    :Component(), m_VoxelDim(voxel_dim), m_Extent(extent), m_VoxelCam(voxel_cam), m_LightInjectionRes(light_injection_res)
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

    m_VoxelCam->MoveTo(glm::vec3(0.0f, m_Extent, 0.0f));
    m_VoxelCam->LookAtDir(glm::vec3(0.0f, -1.0f, 0.0f), glm::vec3(0.0f, 0.0f, 1.0f));
    glm::mat4x4 view_to_down = m_VoxelCam->GetViewMtx();

    m_VoxelCam->MoveTo(glm::vec3(m_Extent, 0.0, 0.0f));
    m_VoxelCam->LookAtDir(glm::vec3(-1.0f, 0.0f, 0.0f), glm::vec3(0.0f, 1.0f, 0.0f));
    glm::mat4x4 view_to_left = m_VoxelCam->GetViewMtx();

    m_VoxelCam->MoveTo(glm::vec3(0.0f, 0.0f, -m_Extent));
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
	//DispatchLightInjection();
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

	glEnable(GL_DEPTH_TEST);
	glEnable(GL_CULL_FACE);
	glPolygonMode(GL_FRONT_AND_BACK, GL_FILL);
	glColorMask(GL_FALSE, GL_FALSE, GL_FALSE, GL_FALSE);
	glDepthMask(GL_FALSE);
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

		glBindBuffer(GL_UNIFORM_BUFFER, 0);

		for (ModelRenderer* renderer : renderers)
			renderer->Render(RenderPass::kLightInjection);
	}
	


	glMemoryBarrier(GL_ALL_BARRIER_BITS);

	glEnable(GL_DEPTH_TEST);
	glEnable(GL_CULL_FACE);
	glColorMask(GL_TRUE, GL_TRUE, GL_TRUE, GL_TRUE);
	glDepthMask(GL_TRUE);
	glViewport(0, 0, vw, vh);
}
