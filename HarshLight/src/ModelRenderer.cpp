#include "ModelRenderer.h"
#include <glm/gtc/type_ptr.hpp>
#include <glm/gtc/matrix_transform.hpp>

ModelRenderer::ModelRenderer(Model * model)
	:m_Model(model), m_Transform(1.0f), m_RenderPassFlag(RenderPass::kNone)
{
#ifdef _DEBUG
    assert(m_Model != nullptr);
#endif
}

void ModelRenderer::MoveTo(const glm::vec3& pos)
{
	m_Transform[3] = glm::vec4(pos, 1.0f);
}

void ModelRenderer::ScaleTo(const glm::vec3& scale)
{
	const float inv_len0 = 1.0f / glm::length(m_Transform[0]);
	m_Transform[0] *= inv_len0 * scale.x;
	const float inv_len1 = 1.0f / glm::length(m_Transform[1]);
	m_Transform[1] *= inv_len1 * scale.y;
	const float inv_len2 = 1.0f / glm::length(m_Transform[2]);
	m_Transform[2] *= inv_len2 * scale.z;
}


void ModelRenderer::Render(RenderPassFlag pass)
{
	switch (pass)
	{
    case RenderPass::kVoxelize:
        if(m_RenderPassFlag & pass) 
            m_Model->Render(m_Transform, m_VoxelizeMaterials); break;
    case RenderPass::kRegular:
        if(m_RenderPassFlag & pass) 
            m_Model->Render(m_Transform, m_Materials); break;
    case RenderPass::kPost:
        if(m_RenderPassFlag & pass) 
            m_Model->Render(m_Transform, m_PostMaterials); break;
    default:case RenderPass::kNone:
		break;
	}
}


void ModelRenderer::AddMaterial(RenderPassFlag pass, const Material* material)
{
#ifdef _DEBUG
    assert(material != nullptr);
#endif
    switch (pass)
    {
    case RenderPass::kVoxelize: m_VoxelizeMaterials.push_back(material); break;
    case RenderPass::kRegular: m_Materials.push_back(material); break;
    case RenderPass::kPost: m_PostMaterials.push_back(material); break;
    default:case RenderPass::kNone:
        break;
    }
}

void ModelRenderer::SetRenderPass(RenderPassFlag flag)
{
	m_RenderPassFlag = flag;
}

RenderPassFlag ModelRenderer::GetRenderPass() const
{
	return m_RenderPassFlag;
}
