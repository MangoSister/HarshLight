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
	glm::mat4x4 m_LightMtx;
	glm::mat4x4 m_LightProjMtx;

	void UpdateLightMtx();
};

struct PointLight
{
public:
	explicit PointLight(const glm::vec3& pos, const glm::vec4& color);

	glm::vec4 m_Position;
	glm::vec4 m_Color;

	void GomputeCubeLightMtx(float near, float far, glm::mat4x4 light_mtx[6], glm::mat4x4& light_proj_mtx) const;
};

class LightManager
{
public:
	LightManager();
	~LightManager();

	void UpdateLight(UniformBufferBinding binding);

	static const uint32_t s_DirLightMaxNum = 4;
	static const uint32_t s_PointLightMaxNum = 4;

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
	void DeleteDirLight(uint32_t idx);
	void DeletePointLight(uint32_t idx);
	void UseDirLight(uint32_t idx);
	void StopUseDirLight(uint32_t idx);
	void UsePointLight(uint32_t idx);
	void StopPointLight(uint32_t idx);

    const glm::vec3& GetAmbient() const;
    const DirLight& GetDirLight(uint32_t idx) const;
    DirLight& GetDirLight(uint32_t idx);
	uint8_t GetDirLightUsage(uint32_t idx) const;
    const PointLight& GetPointLight(uint32_t idx) const;
    PointLight& GetPointLight(uint32_t idx);
	uint8_t GetPointLightUsage(uint32_t idx) const;

	uint32_t GetActiveDirLightCount() const;
	uint32_t GetActivePointLightCount() const;
    uint32_t GetDirLightCount() const;
    uint32_t GetPointLightCount() const;



    const glm::vec3& GetPointLightAtten() const;
    void SetPointLightAtten(const glm::vec3& atten);

    float ComputePointLightCutoffRadius(const PointLight& light, float atten) const;

private:
	std::vector<DirLight> m_DirLights;
	std::vector<uint8_t> m_DirLightsUsage;
	uint32_t m_ActiveDirLightsCount = 0;

	std::vector<PointLight> m_PointLights;
	std::vector<uint8_t> m_PointLightsUsage;
	uint32_t m_ActivePointLightsCount = 0;

    glm::vec3 m_Ambient;

    glm::vec3 m_PointLightAtten;

	GLuint m_LightUBuffer;
};