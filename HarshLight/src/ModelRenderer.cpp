#include "ModelRenderer.h"
#include <glm/gtc/type_ptr.hpp>
#include <glm/gtc/matrix_transform.hpp>

ModelRenderer::ModelRenderer(Model * model, Material * material)
	:Component(),
	m_Model(model), m_Material(material), m_Transform(1.0f)
{
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
	m_Material->Use();
	GLuint model_loc = glGetUniformLocation(m_Material->GetProgram(), "model");
#if _DEBUG
	if (model_loc == GL_INVALID_VALUE)
		fprintf(stderr, "WARNING: Invalid model mtx shader program location\n");
#endif
	glUniformMatrix4fv(model_loc, 1, GL_FALSE, glm::value_ptr(m_Transform));
    m_Model->Render(m_Material);
}
