#include <glm/gtc/type_ptr.hpp>
#include "Light.h"

DirLight::DirLight(const glm::vec3& direction, const glm::vec4& color) : m_Direction(direction.x, direction.y, direction.z, 0.0f), m_Color(color) { }
PointLight::PointLight(const glm::vec3& pos, const glm::vec4& color) : m_Position(pos.x, pos.y, pos.z, 1.0f), m_Color(color) { }

LightManager::LightManager() : m_LightUBuffer(0)
{
	glGenBuffers(1, &m_LightUBuffer);
	glBindBuffer(GL_UNIFORM_BUFFER, m_LightUBuffer);
	glBufferData(GL_UNIFORM_BUFFER, GetLightUBufferSize(), nullptr, GL_STATIC_DRAW);

    //uint32_t offset = 0;
    //uint32_t dir_light_num = 0;
    //glBufferSubData(GL_UNIFORM_BUFFER, offset, sizeof(uint32_t), &dir_light_num);
    //offset += sizeof(uint32_t);

    //uint32_t pt_light_num = 0;
    //glBufferSubData(GL_UNIFORM_BUFFER, offset, sizeof(uint32_t), &pt_light_num);

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
    //glBindBuffer(GL_UNIFORM_BUFFER, m_LightUBuffer);
    uint32_t offset = 0;
    for (uint32_t i = 0; i < m_DirLights.size(); i++)
    {
        glBufferSubData(GL_UNIFORM_BUFFER, offset, sizeof(DirLight), &m_DirLights[i]);
        offset += sizeof(DirLight);
    }

    offset = s_DirLightMaxNum * sizeof(DirLight);
    for (uint32_t i = 0; i < m_PointLights.size(); i++)
    {
        glBufferSubData(GL_UNIFORM_BUFFER, offset, sizeof(PointLight), &m_PointLights[i]);
        offset += sizeof(PointLight);
    }

    offset = s_DirLightMaxNum * sizeof(DirLight) + s_PointLightMaxNum * sizeof(PointLight);
    glBufferSubData(GL_UNIFORM_BUFFER, offset, sizeof(glm::vec4), glm::value_ptr(glm::vec4(m_Ambient, 1.0f)));
    offset += sizeof(glm::vec4);

    glBufferSubData(GL_UNIFORM_BUFFER, offset, sizeof(glm::vec4), glm::value_ptr(glm::vec4(m_PointLightAtten, 0.0f)));
    offset += sizeof(glm::vec4);

    uint32_t dir_light_num = static_cast<uint32_t>(m_DirLights.size());
    glBufferSubData(GL_UNIFORM_BUFFER, offset, sizeof(uint32_t), &dir_light_num);
    offset += sizeof(uint32_t);

    uint32_t pt_light_num = static_cast<uint32_t>(m_PointLights.size());
    glBufferSubData(GL_UNIFORM_BUFFER, offset, sizeof(uint32_t), &pt_light_num);
    offset += sizeof(uint32_t);

	glBindBuffer(GL_UNIFORM_BUFFER, 0);
}

void LightManager::SetAmbient(const glm::vec3& ambient)
{
    m_Ambient = ambient;
}

void LightManager::AddDirLight(const DirLight& light)
{
    if (m_DirLights.size() < s_DirLightMaxNum)
        m_DirLights.push_back(light);
}

void LightManager::AddPointLight(const PointLight& light)
{
    if (m_PointLights.size() < s_PointLightMaxNum)
        m_PointLights.push_back(light);
}

const glm::vec3& LightManager::GetAmbient() const
{
    return m_Ambient;
}

const DirLight& LightManager::GetDirLight(uint32_t idx) const
{
#ifdef _DEBUG
    assert(idx >= 0 && idx < m_DirLights.size());
#endif

    return m_DirLights[idx];
}

DirLight& LightManager::GetDirLight(uint32_t idx)
{
#ifdef _DEBUG
    assert(idx >= 0 && idx < m_DirLights.size());
#endif

    return m_DirLights[idx];
}

const PointLight& LightManager::GetPointLight(uint32_t idx) const
{
#ifdef _DEBUG
    assert(idx >= 0 && idx < m_PointLights.size());
#endif

    return m_PointLights[idx];
}

PointLight& LightManager::GetPointLight(uint32_t idx)
{
#ifdef _DEBUG
    assert(idx >= 0 && idx < m_PointLights.size());
#endif

    return m_PointLights[idx];
}

uint32_t LightManager::GetDirLightCount() const
{
    return m_DirLights.size();
}

uint32_t LightManager::GetPointLightCount() const
{
    return m_PointLights.size();
}

void LightManager::DeleteDirLight(uint32_t idx)
{
#ifdef _DEBUG
    assert(idx >= 0 && idx < m_DirLights.size());
#endif

    m_DirLights.erase(m_DirLights.begin() + idx);
}

void LightManager::DeletePointLight(uint32_t idx)
{
#ifdef _DEBUG
    assert(idx >= 0 && idx < m_PointLights.size());
#endif

    m_PointLights.erase(m_PointLights.begin() + idx);
}

const glm::vec3 & LightManager::GetPointLightAtten() const
{
    return m_PointLightAtten;
}

void LightManager::SetPointLightAtten(const glm::vec3& atten)
{
    m_PointLightAtten = atten;
}
