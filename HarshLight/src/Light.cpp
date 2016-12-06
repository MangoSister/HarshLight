#include <glm/glm.hpp>
#include <glm/gtc/type_ptr.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include "Light.h"

DirLight::DirLight(const glm::vec3& direction, const glm::vec4& color) : m_Color(color), m_LightMtx(1.0f) 
{
    glm::vec3 dir = glm::normalize(direction);
    m_Direction = glm::vec4(dir.x, dir.y, dir.z, 0.0f);
}

void DirLight::UpdateLightMtx()
{
	glm::vec3 up(0.0f, 1.0f, 0.0f);
	if (m_Direction.y == 1.0f)
		up = glm::vec3(0.0f, 0.0f, -1.0f);
	else if(m_Direction.y == -1.0f)
		up = glm::vec3(0.0f, 0.0f, 1.0f);

	glm::vec3 dir(m_Direction.x, m_Direction.y, m_Direction.z);

	const glm::vec4 r = glm::vec4(normalize(cross(up, dir)), 0.0f);
	const glm::vec4 u = glm::vec4(normalize(cross(dir, glm::vec3(r))), 0.0f);
	const glm::vec4 f = glm::vec4(-normalize(dir), 0.0f);

	m_LightMtx[0][0] = r[0];  m_LightMtx[0][1] = u[0];  m_LightMtx[0][2] = f[0];
	m_LightMtx[1][0] = r[1];  m_LightMtx[1][1] = u[1];  m_LightMtx[1][2] = f[1];
	m_LightMtx[2][0] = r[2];  m_LightMtx[2][1] = u[2];  m_LightMtx[2][2] = f[2];
}
PointLight::PointLight(const glm::vec3& pos, const glm::vec4& color) : m_Position(pos.x, pos.y, pos.z, 1.0f), m_Color(color) { }

void PointLight::GomputeCubeLightMtx(float near, float far, glm::mat4x4 light_mtx[6], glm::mat4x4& light_proj_mtx) const
{
	light_proj_mtx = glm::perspective(glm::radians(90.0f), 1.0f, near, far);
	
    glm::vec3 pos(m_Position);

    light_mtx[0] = LookAtDir(glm::vec3(1.0f, 0.0f, 0.0f), glm::vec3(0.0f, 1.0f, 0.0f));
    light_mtx[1] = LookAtDir(glm::vec3(-1.0f, 0.0f, 0.0f), glm::vec3(0.0f, 1.0f, 0.0f));

    light_mtx[2] = LookAtDir(glm::vec3(0.0f, 1.0f, 0.0f), glm::vec3(0.0f, 0.0f, -1.0f));
    light_mtx[3] = LookAtDir(glm::vec3(0.0f, -1.0f, 0.0f), glm::vec3(0.0f, 0.0f, 1.0f));

    light_mtx[4] = LookAtDir(glm::vec3(0.0f, 0.0f, 1.0f), glm::vec3(0.0f, 1.0f, 0.0f));
    light_mtx[5] = LookAtDir(glm::vec3(0.0f, 0.0f, -1.0f), glm::vec3(0.0f, 1.0f, 0.0f));

    for (uint32_t i = 0; i < 6; i++)
    {
        light_mtx[i][3] = m_Position;
        light_mtx[i] = ViewMtxFromTransform(light_mtx[i]);
    }


    //light_mtx[0] = glm::lookAt(pos, pos + glm::vec3(1.0, 0.0, 0.0), glm::vec3(0.0, 1.0, 0.0));
    //light_mtx[1] = glm::lookAt(pos, pos + glm::vec3(-1.0, 0.0, 0.0), glm::vec3(0.0, 1.0, 0.0));
    //light_mtx[2] = glm::lookAt(pos, pos + glm::vec3(0.0, 1.0, 0.0), glm::vec3(0.0, 0.0, 1.0));
    //light_mtx[3] = glm::lookAt(pos, pos + glm::vec3(0.0, -1.0, 0.0), glm::vec3(0.0, 0.0, -1.0));
    //light_mtx[4] = glm::lookAt(pos, pos + glm::vec3(0.0, 0.0, 1.0), glm::vec3(0.0, 1.0, 0.0));
    //light_mtx[5] = glm::lookAt(pos, pos + glm::vec3(0.0, 0.0, -1.0), glm::vec3(0.0, 1.0, 0.0));
}

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
	for (uint32_t i = 0; i < m_DirLights.size(); i++)
		m_DirLights[i].UpdateLightMtx();
	
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
	return static_cast<uint32_t>(m_DirLights.size());
}

uint32_t LightManager::GetPointLightCount() const
{
    return static_cast<uint32_t>(m_PointLights.size());
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

float LightManager::ComputePointLightCutoffRadius(const PointLight & light, float atten) const
{
    return std::sqrt(light.m_Color.w / (atten * m_PointLightAtten.z));
}
