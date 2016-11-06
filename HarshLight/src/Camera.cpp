#include <glm/gtc/matrix_transform.hpp>
#include "World.h"
#include "Camera.h"

Camera::Camera(float fovY, float aspect, float near, float far)
	:Component(),
    m_ViewMtx(1.0f), m_FovY(fovY), m_Aspect(aspect), m_Near(near), m_Far(far)
{
   m_ProjMtx = glm::perspective(fovY, aspect, near, far);   
}

Camera::~Camera()
{

}

void Camera::Start()
{
    
}

void Camera::Update(float dt)
{
	if (World::GetInst().GetKey(GLFW_KEY_W) == GLFW_PRESS)
	{
		glm::translate(m_ViewMtx, vec3(0.0f, 0.0f, m_PanningSpeed * dt));
	}

	if (World::GetInst().GetKey(GLFW_KEY_S) == GLFW_PRESS)
	{
		glm::translate(m_ViewMtx, vec3(0.0f, 0.0f, m_PanningSpeed  * -dt));
	}

	if (World::GetInst().GetKey(GLFW_KEY_A) == GLFW_PRESS)
	{
		glm::translate(m_ViewMtx, vec3(m_PanningSpeed * -dt, 0.0f, 0.0f));
	}

	if (World::GetInst().GetKey(GLFW_KEY_D) == GLFW_PRESS)
	{
		glm::translate(m_ViewMtx, vec3(m_PanningSpeed * dt, 0.0f, 1.0f));
	}

	if (World::GetInst().GetKey(GLFW_KEY_Q) == GLFW_PRESS)
	{
		glm::translate(m_ViewMtx, vec3(0.0f, m_PanningSpeed * dt, 0.0f));
	}

	if (World::GetInst().GetKey(GLFW_KEY_E) == GLFW_PRESS)
	{
		glm::translate(m_ViewMtx, vec3(0.0f, m_PanningSpeed  * -dt, 0.0f));
	}

}

void Camera::LookAt(const vec3& target, const vec3& up)
{

}