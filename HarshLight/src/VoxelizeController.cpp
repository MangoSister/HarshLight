#include "VoxelizeController.h"
#include "ModelRenderer.h"
#include "Actor.h"

const char* VoxelizeController::s_VoxelDimName = "VoxelDim";

VoxelizeController::VoxelizeController(uint32_t dim)
    :Component(), m_VoxelDim(dim)
{ }

void VoxelizeController::Start()
{
	const std::vector<Material*>& voxel_mats = m_Actor->GetRenderer<ModelRenderer>()->GetMaterial(RenderPass::kVoxelize);
	for (auto mat : voxel_mats)
	{
		if (mat)
			mat->SetVec2Param(s_VoxelDimName, glm::vec2(static_cast<float>(m_VoxelDim), static_cast<float>(m_VoxelDim)));
	}

	const std::vector<Material*>& regular_mats = m_Actor->GetRenderer<ModelRenderer>()->GetMaterial(RenderPass::kRegular);
	for (auto mat : regular_mats)
	{
		if (mat)
			mat->SetVec2Param(s_VoxelDimName, glm::vec2(static_cast<float>(m_VoxelDim), static_cast<float>(m_VoxelDim)));
	}
}

void VoxelizeController::SetVoxelDim(uint32_t dim)
{
    m_VoxelDim = dim;
}

uint32_t VoxelizeController::GetVoxelDim() const
{
    return m_VoxelDim;
}
