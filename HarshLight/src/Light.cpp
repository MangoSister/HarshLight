#include "Light.h"

DirLight::DirLight(const glm::vec3& direction, const glm::vec3& color) : m_Direction(direction), m_Color(color) { }
PointLight::PointLight(const glm::vec3& pos, const glm::vec4& coefs, const glm::vec3& color) : m_Position(pos), m_Coefs(coefs), m_Color(color) { }

LightManager::LightManager() : m_LightUBuffer(0)
{
	glGenBuffers(1, &m_LightUBuffer);
	glBindBuffer(GL_UNIFORM_BUFFER, m_LightUBuffer);
	glBufferData(GL_UNIFORM_BUFFER, GetLightUBufferSize(), nullptr, GL_STATIC_DRAW);
	glBindBuffer(GL_UNIFORM_BUFFER, 0);
}

LightManager::~LightManager()
{
	if (m_LightUBuffer)
	{
		glDeleteBuffers(1, &m_LightUBuffer);
		m_LightUBuffer = 0;
	}
}

void LightManager::UpdateLight(UniformBufferBinding binding)
{
	glBindBufferRange(GL_UNIFORM_BUFFER, (uint8_t)binding, m_LightUBuffer, 0, GetLightUBufferSize());
	glBindBuffer(GL_UNIFORM_BUFFER, m_LightUBuffer);

	for (uint32_t i = 0; i < m_DirLights.size(); i++)
		glBufferSubData(GL_UNIFORM_BUFFER, i * sizeof(DirLight), sizeof(DirLight), &m_DirLights[i]);

	for (uint32_t i = 0; i < m_PointLights.size(); i++)
		glBufferSubData(GL_UNIFORM_BUFFER, s_DirLightMaxNum * sizeof(DirLight) + i * sizeof(PointLight), sizeof(PointLight), &m_PointLights[i]);

	glBindBuffer(GL_UNIFORM_BUFFER, 0);
}
