#include "VoxelizeController.h"
#include "ModelRenderer.h"
#include "Actor.h"
#include "World.h"

const char* VoxelizeController::s_VoxelDimName = "VoxelDim";
const char* VoxelizeController::s_ViewMtxToDownName = "ViewMtxToDown";
const char* VoxelizeController::s_ViewMtxToLeftName = "ViewMtxToLeft";
const char* VoxelizeController::s_ViewMtxToForwardName = "ViewMtxToForward";

const char* VoxelizeController::s_VoxelChannelNames[s_VoxelChannelNum]
{
	"Albedo",
	"Normal",
};

VoxelizeController::VoxelizeController(uint32_t dim, float extent, Camera* voxel_cam)
    :Component(), m_VoxelDim(dim), m_Extent(extent), m_VoxelCam(voxel_cam)
{
	for (uint32_t i = 0; i < s_VoxelChannelNum; i++)
		m_VoxelizeTex[i] = new Texture3dCompute(dim, dim, dim, GL_R32UI, GL_RED_INTEGER, GL_UNSIGNED_INT);
}

VoxelizeController::~VoxelizeController()
{
	//unregister voxel_tex from cuda
	for (uint32_t i = 0; i < s_VoxelChannelNum; i++)
	{
		cudaCheckError(cudaGraphicsUnregisterResource(m_CudaResources[i]));
		if (m_VoxelizeTex[i])
		{
			delete m_VoxelizeTex[i];
			m_VoxelizeTex[i] = nullptr;
		}
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
	for (uint32_t i = 0; i < s_VoxelChannelNum; i++)
	{
		cudaCheckError(cudaGraphicsGLRegisterImage(&m_CudaResources[i], m_VoxelizeTex[i]->GetTexObj(),
			GL_TEXTURE_3D, cudaGraphicsRegisterFlagsSurfaceLoadStore));
	}

	DispatchVoxelization();
}

void VoxelizeController::SetVoxelDim(uint32_t dim)
{
    m_VoxelDim = dim;
}

uint32_t VoxelizeController::GetVoxelDim() const
{
    return m_VoxelDim;
}

void VoxelizeController::TransferVoxelDataToCuda(cudaSurfaceObject_t surf_objs[s_VoxelChannelNum])
{
	cudaCheckError(cudaGraphicsMapResources(s_VoxelChannelNum, m_CudaResources, 0));
	
	for (uint32_t i = 0; i < s_VoxelChannelNum; i++)
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

void VoxelizeController::FinishVoxelDataFromCuda(cudaSurfaceObject_t surf_objs[s_VoxelChannelNum])
{
    //there is no unbinding surface API
	for (uint32_t i = 0; i < s_VoxelChannelNum; i++)
		cudaCheckError(cudaDestroySurfaceObject(surf_objs[i]));

	cudaCheckError(cudaGraphicsUnmapResources(s_VoxelChannelNum, m_CudaResources, 0));
}

void VoxelizeController::DispatchVoxelization()
{
	for (uint32_t i = 0; i < s_VoxelChannelNum; i++)
		if (m_VoxelizeTex[i] == nullptr) return;
	for (uint32_t i = 0; i < s_VoxelChannelNum; i++)
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
