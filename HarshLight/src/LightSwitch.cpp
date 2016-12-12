#include "LightSwitch.h"
#include "World.h"
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <cmath>

void LightSwitch::Start()
{
	if (!World::GetInst().GetLightManager().GetDirLightCount())
		return;

	DirLight& dir_light = World::GetInst().GetLightManager().GetDirLight(0);
	m_CurrentPitchDeg = glm::degrees(asinf(dir_light.m_Direction.y));
	m_CurrentYawDeg = glm::degrees(atan2f(dir_light.m_Direction.z, dir_light.m_Direction.x));
}

void LightSwitch::Update(float dt)
{
	if (World::GetInst().GetLightManager().GetActiveDirLightCount())
	{
		DirLight& dir_light = World::GetInst().GetLightManager().GetDirLight(0);
		const float rot_speed = 20.0f;
		if (World::GetInst().IsKeyDown(GLFW_KEY_I))
		{
			m_CurrentPitchDeg += dt * rot_speed;
		}

		if (World::GetInst().IsKeyDown(GLFW_KEY_K))
		{
			m_CurrentPitchDeg -= dt * rot_speed;
		}

		if (World::GetInst().IsKeyDown(GLFW_KEY_J))
		{
			m_CurrentYawDeg += dt * rot_speed;
		}

		if (World::GetInst().IsKeyDown(GLFW_KEY_L))
		{
			m_CurrentYawDeg -= dt * rot_speed;
		}

		m_CurrentYawDeg = fmod(m_CurrentYawDeg, 360.0f);
		m_CurrentPitchDeg = std::fmin(std::fmax(m_CurrentPitchDeg, -89.0f), 89.0f);

		glm::vec3 next_dir;
		next_dir.z = std::cosf(glm::radians(m_CurrentPitchDeg)) * std::cosf(glm::radians(m_CurrentYawDeg));
		next_dir.y = std::sinf(glm::radians(m_CurrentPitchDeg));
		next_dir.x = std::cosf(glm::radians(m_CurrentPitchDeg)) * std::sinf(glm::radians(m_CurrentYawDeg));
		next_dir = normalize(next_dir);
		dir_light.m_Direction = glm::vec4(next_dir, 1.0f);
	}

	if (World::GetInst().GetLightManager().GetPointLightCount())
	{	
		if (World::GetInst().GetKey(GLFW_KEY_F) == GLFW_PRESS)
		{
			if (World::GetInst().GetLightManager().GetPointLightUsage(0))
				World::GetInst().GetLightManager().StopPointLight(0);
			else World::GetInst().GetLightManager().UsePointLight(0);
		}
	}
	
}
