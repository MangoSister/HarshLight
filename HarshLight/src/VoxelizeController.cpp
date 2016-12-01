#include "VoxelizeController.h"
#include "ModelRenderer.h"
#include "Actor.h"
#include "World.h"

const char* VoxelizeController::s_VoxelDimName = "VoxelDim";
const char* VoxelizeController::s_ViewMtxToDownName = "ViewMtxToDown";
const char* VoxelizeController::s_ViewMtxToLeftName = "ViewMtxToLeft";
const char* VoxelizeController::s_ViewMtxToForwardName = "ViewMtxToForward";

VoxelizeController::VoxelizeController(uint32_t dim, float extent, Camera* voxel_cam, Texture3dCompute* voxel_tex)
    :Component(), m_VoxelDim(dim), m_Extent(extent), m_VoxelCam(voxel_cam), m_VoxelizeTex(voxel_tex)
{ }

VoxelizeController::~VoxelizeController()
{
	//unregister voxel_tex from cuda
	cudaCheckError(cudaGraphicsUnregisterResource(m_CudaResource));

	if (m_VoxelizeTex)
	{
		delete m_VoxelizeTex;
		m_VoxelizeTex = nullptr;
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
	
	cudaCheckError(cudaGraphicsGLRegisterImage(&m_CudaResource, m_VoxelizeTex->GetTexObj(), 
					GL_TEXTURE_3D, cudaGraphicsRegisterFlagsSurfaceLoadStore));

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

cudaSurfaceObject_t VoxelizeController::TransferVoxelDataToCuda()
{
	cudaCheckError(cudaGraphicsMapResources(1, &m_CudaResource, 0));
	cudaArray* voxel_cuda_array;
	cudaCheckError(cudaGraphicsSubResourceGetMappedArray(&voxel_cuda_array, m_CudaResource, 0, 0));

    struct cudaResourceDesc res_desc;
    memset(&res_desc, 0, sizeof(cudaResourceDesc));
    res_desc.resType = cudaResourceTypeArray;    // be sure to set the resource type to cudaResourceTypeArray
    res_desc.res.array.array = voxel_cuda_array;

    cudaSurfaceObject_t surf_obj = 0;
    cudaCheckError(cudaCreateSurfaceObject(&surf_obj, &res_desc));
	//cudaCheckError(cudaBindSurfaceToArray(surfaceWrite, cuda_array));

    return surf_obj;
}

void VoxelizeController::FinishVoxelDataFromCuda(cudaSurfaceObject_t surf_obj)
{
    //there is no unbinding surface API

    cudaCheckError(cudaDestroySurfaceObject(surf_obj));
    cudaCheckError(cudaGraphicsUnmapResources(1, &m_CudaResource, 0));
}

void VoxelizeController::DispatchVoxelization()
{
	if (m_VoxelizeTex)
	{
		m_VoxelizeTex->CleanContent();

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
			glMemoryBarrier(GL_ALL_BARRIER_BITS);
		}

		glEnable(GL_DEPTH_TEST);
		glEnable(GL_CULL_FACE);
		glColorMask(GL_TRUE, GL_TRUE, GL_TRUE, GL_TRUE);
		glDepthMask(GL_TRUE);
		uint32_t vw, vh;
		World::GetInst().GetViewportSize(vw, vh);
		glViewport(0, 0, vw, vh);

	}
}
