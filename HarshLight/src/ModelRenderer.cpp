#include "ModelRenderer.h"
#include <glm/gtc/type_ptr.hpp>
#include <glm/gtc/matrix_transform.hpp>

ModelRenderer::ModelRenderer(Model * model)
	:Component(),
	m_Model(model), m_Transform(1.0f)
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
	const float inv_len0 = 1.0f / m_Transform[0].length();
	m_Transform[0] *= inv_len0 * scale.x;
	const float inv_len1 = 1.0f / m_Transform[1].length();
	m_Transform[1] *= inv_len1 * scale.x;
	const float inv_len2 = 1.0f / m_Transform[2].length();
	m_Transform[2] *= inv_len2 * scale.x;
}

void ModelRenderer::Start()
{

}

void ModelRenderer::Update(float dt)
{
#ifdef _DEBUG
    assert(m_Materials.size() > 0);
#endif
    m_Model->Render(m_Transform, m_Materials);
}

void ModelRenderer::AddMaterial(const Material * material)
{
#ifdef _DEBUG
    assert(material != nullptr);
#endif
    m_Materials.push_back(material);
}