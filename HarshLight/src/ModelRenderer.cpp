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
	case RenderPass::kDirLightInjection:
		if (m_RenderPassFlag & pass)
			m_Model->Render(m_Transform, m_DirLightInjectionMaterials); break;
    case RenderPass::kPointLightInjection:
        if (m_RenderPassFlag & pass)
            m_Model->Render(m_Transform, m_PointLightInjectionMaterials); break;
    case RenderPass::kGeometry:
        if(m_RenderPassFlag & pass) 
            m_Model->Render(m_Transform, m_GeometryPassMaterial); break;
	case RenderPass::kDeferredIndirectDiffuse:
		if (m_RenderPassFlag & pass)
			m_Model->Render(m_Transform, m_DeferredIndirectDiffuseMaterial); break;
	case RenderPass::kDeferredFinalComposition:
		if (m_RenderPassFlag & pass)
			m_Model->Render(m_Transform, m_DeferredFinalComposition); break;
    case RenderPass::kPost:
        if(m_RenderPassFlag & pass) 
            m_Model->Render(m_Transform, m_PostMaterials); break;
    default:case RenderPass::kNone:
		break;
	}
}

void ModelRenderer::Render(RenderPassFlag pass, const glm::vec3 & center, float radius)
{
    switch (pass)
    {
    case RenderPass::kVoxelize:
        if (m_RenderPassFlag & pass)
            m_Model->Render(m_Transform, m_VoxelizeMaterials, center, radius); break;
    case RenderPass::kDirLightInjection:
        if (m_RenderPassFlag & pass)
            m_Model->Render(m_Transform, m_DirLightInjectionMaterials, center, radius); break;
    case RenderPass::kPointLightInjection:
        if (m_RenderPassFlag & pass)
            m_Model->Render(m_Transform, m_PointLightInjectionMaterials, center, radius); break;
    case RenderPass::kGeometry:
        if (m_RenderPassFlag & pass)
            m_Model->Render(m_Transform, m_GeometryPassMaterial, center, radius); break;
    case RenderPass::kDeferredIndirectDiffuse:
        if (m_RenderPassFlag & pass)
            m_Model->Render(m_Transform, m_DeferredIndirectDiffuseMaterial, center, radius); break;
	case RenderPass::kDeferredFinalComposition:
		if (m_RenderPassFlag & pass)
			m_Model->Render(m_Transform, m_DeferredFinalComposition, center, radius); break;
    case RenderPass::kPost:
        if (m_RenderPassFlag & pass)
            m_Model->Render(m_Transform, m_PostMaterials, center, radius); break;
    default:case RenderPass::kNone:
        break;
    }
}


void ModelRenderer::AddMaterial(RenderPassFlag pass, Material* material)
{
#ifdef _DEBUG
    assert(material != nullptr);
#endif
    switch (pass)
    {
    case RenderPass::kVoxelize: m_VoxelizeMaterials.push_back(material); break;
	case RenderPass::kDirLightInjection: m_DirLightInjectionMaterials.push_back(material); break;
    case RenderPass::kPointLightInjection:m_PointLightInjectionMaterials.push_back(material); break;
    case RenderPass::kGeometry: m_GeometryPassMaterial.push_back(material); break;
    case RenderPass::kDeferredIndirectDiffuse: m_DeferredIndirectDiffuseMaterial.push_back(material); break;
	case RenderPass::kDeferredFinalComposition: m_DeferredFinalComposition.push_back(material); break;
    case RenderPass::kPost: m_PostMaterials.push_back(material); break;
    default:case RenderPass::kNone:
        break;
    }
}

const std::vector<Material*>& ModelRenderer::GetMaterial(RenderPassFlag pass) const
{
    switch (pass)
    {
    case RenderPass::kVoxelize: return m_VoxelizeMaterials;
	case RenderPass::kDirLightInjection: return m_DirLightInjectionMaterials;
    case RenderPass::kPointLightInjection: return m_PointLightInjectionMaterials;
    case RenderPass::kGeometry:default: return m_GeometryPassMaterial;
    case RenderPass::kDeferredIndirectDiffuse: return m_DeferredIndirectDiffuseMaterial;
	case RenderPass::kDeferredFinalComposition: return m_DeferredFinalComposition;
    case RenderPass::kPost: return m_PostMaterials;
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

const mat4x4 & ModelRenderer::GetTransform() const
{
    return m_Transform;
}
