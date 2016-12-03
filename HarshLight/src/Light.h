#pragma once
#include "Util.h"
#include <vector>
#include <glm/glm.hpp>

struct DirLight 
{
public:
	explicit DirLight(const glm::vec3& direction, const glm::vec4& color);

	glm::vec4 m_Direction;
	glm::vec4 m_Color;
};

struct PointLight
{
public:
	explicit PointLight(const glm::vec3& pos, const glm::vec4& color);

	glm::vec4 m_Position;
	//glm::vec4 m_Coefs;
	glm::vec4 m_Color;
};

class LightManager
{
public:
	LightManager();
	~LightManager();

	void UpdateLight(UniformBufferBinding binding);

	static const uint32_t s_DirLightMaxNum = 4;
	static const uint32_t s_PointLightMaxNum = 8;
    static constexpr uint32_t GetLightUBufferSize()
    {
        return s_DirLightMaxNum * sizeof(DirLight) +
            s_PointLightMaxNum * sizeof(PointLight) +
            2 * sizeof(uint32_t) +
            2 * sizeof(glm::vec4);
    }

    void SetAmbient(const glm::vec3& ambient);
    void AddDirLight(const DirLight& light);
    void AddPointLight(const PointLight& light);

    const glm::vec3& GetAmbient() const;
    const DirLight& GetDirLight(uint32_t idx) const;
    DirLight& GetDirLight(uint32_t idx);
    const PointLight& GetPointLight(uint32_t idx) const;
    PointLight& GetPointLight(uint32_t idx);

    uint32_t GetDirLightCount() const;
    uint32_t GetPointLightCount() const;

    void DeleteDirLight(uint32_t idx);
    void DeletePointLight(uint32_t idx);

    const glm::vec3& GetPointLightAtten() const;
    void SetPointLightAtten(const glm::vec3& atten);

private:
	std::vector<DirLight> m_DirLights;
	std::vector<PointLight> m_PointLights;
    glm::vec3 m_Ambient;

    glm::vec3 m_PointLightAtten;

	GLuint m_LightUBuffer;
};