#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/type_ptr.hpp>
#include "World.h"
#include "Camera.h"
#include "Util.h"

Camera::Camera(float fovY, float aspect, float near, float far)
	:Component(),
    m_ViewMtx(1.0f), m_FovY(fovY), m_Aspect(aspect), m_Near(near), m_Far(far), m_CamUniformBuffer(0), m_PanningSpeed(10.0f)
{
   m_ProjMtx = glm::perspective(fovY, aspect, near, far);   
   
   //create camera unifrom buffer
   glGenBuffers(1, &m_CamUniformBuffer);
   glBindBuffer(GL_UNIFORM_BUFFER, m_CamUniformBuffer);
   glBufferData(GL_UNIFORM_BUFFER, 2 * sizeof(glm::mat4), nullptr, GL_STATIC_DRAW);
   glBindBuffer(GL_UNIFORM_BUFFER, 0);
   glBindBufferRange(GL_UNIFORM_BUFFER, BINDING_POINT_CAMMTX, m_CamUniformBuffer, 0, 2 * sizeof(glm::mat4));
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
		m_ViewMtx = glm::translate(m_ViewMtx, GetForward() * m_PanningSpeed * -dt);
	}

	if (World::GetInst().GetKey(GLFW_KEY_S) == GLFW_PRESS)
	{
		m_ViewMtx = glm::translate(m_ViewMtx, GetForward() * m_PanningSpeed * dt);
	}

	if (World::GetInst().GetKey(GLFW_KEY_A) == GLFW_PRESS)
	{
		m_ViewMtx = glm::translate(m_ViewMtx, GetRight() * m_PanningSpeed * -dt);
	}

	if (World::GetInst().GetKey(GLFW_KEY_D) == GLFW_PRESS)
	{
		m_ViewMtx = glm::translate(m_ViewMtx, GetRight() * m_PanningSpeed * dt);
	}

	if (World::GetInst().GetKey(GLFW_KEY_Q) == GLFW_PRESS)
	{
		m_ViewMtx = glm::translate(m_ViewMtx, GetUp() * m_PanningSpeed * -dt);
	}

	if (World::GetInst().GetKey(GLFW_KEY_E) == GLFW_PRESS)
	{
		m_ViewMtx = glm::translate(m_ViewMtx, GetUp() * m_PanningSpeed * dt);
	}
}

void Camera::UpdateCamMtx() const
{
	glBindBuffer(GL_UNIFORM_BUFFER, m_CamUniformBuffer);
	glBufferSubData(GL_UNIFORM_BUFFER, 0, sizeof(glm::mat4), glm::value_ptr(m_ViewMtx));
	glBufferSubData(GL_UNIFORM_BUFFER, sizeof(glm::mat4), sizeof(glm::mat4), glm::value_ptr(m_ProjMtx));
	glBindBuffer(GL_UNIFORM_BUFFER, 0);
}

glm::vec3 Camera::GetRight() const
{
	return glm::vec3(m_ViewMtx[0][0], m_ViewMtx[1][0], m_ViewMtx[2][0]);
}

glm::vec3 Camera::GetUp() const
{
	return glm::vec3(m_ViewMtx[0][1], m_ViewMtx[1][1], m_ViewMtx[2][1]);
}

glm::vec3 Camera::GetForward() const
{
	return glm::vec3(m_ViewMtx[0][2], m_ViewMtx[1][2], m_ViewMtx[2][2]);
}

glm::vec3 Camera::GetPos() const
{
	return m_ViewMtx[3];
}

void Camera::MoveTo(const vec3& pos)
{
	m_ViewMtx[3] = glm::vec4(-pos, 1.0f);
}

void Camera::LookAt(const vec3& target, const vec3& up)
{
	m_ViewMtx = glm::lookAt(glm::vec3(m_ViewMtx[3]), target, up);
	
}

void Camera::LookAtDir(const vec3& dir, const vec3 & up)
{
	const glm::vec3 r = glm::vec4(glm::normalize(glm::cross(dir, up)), 0.0f);
	const glm::vec3 u = glm::vec4(glm::normalize(glm::cross(GetRight(), dir)), 0.0f);
	const glm::vec3 f = glm::vec4(dir, 0.0f);
	
	m_ViewMtx[0][0] = r.x;
	m_ViewMtx[1][0] = r.y;
	m_ViewMtx[2][0] = r.z;

	m_ViewMtx[0][1] = u.x;
	m_ViewMtx[1][1] = u.y;
	m_ViewMtx[2][1] = u.z;

	m_ViewMtx[0][2] = f.x;
	m_ViewMtx[1][2] = f.y;
	m_ViewMtx[2][2] = f.z;
}

void Camera::Rotate(const vec3 & axis, float angle_rad)
{
	m_ViewMtx = glm::rotate(m_ViewMtx, angle_rad, axis);
}
