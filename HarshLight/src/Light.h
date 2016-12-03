#pragma once
#include "Util.h"
#include <vector>
#include <glm/glm.hpp>

struct DirLight 
{
public:
	explicit DirLight(const glm::vec3& direction, const glm::vec3& color);

	glm::vec3 m_Direction;
	glm::vec3 m_Color;
};

struct PointLight
{
public:
	explicit PointLight(const glm::vec3& pos, const glm::vec4& coefs, const glm::vec3& color);

	glm::vec3 m_Position;
	glm::vec4 m_Coefs;
	glm::vec3 m_Color;
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
	{ return s_DirLightMaxNum * sizeof(DirLight) + s_PointLightMaxNum * sizeof(PointLight); }

private:
	std::vector<DirLight> m_DirLights;
	std::vector<PointLight> m_PointLights;

	GLuint m_LightUBuffer;
};