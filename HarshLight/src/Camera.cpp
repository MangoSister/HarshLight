#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/type_ptr.hpp>
#include "World.h"
#include "Camera.h"
#include "Util.h"

Camera::Camera(float fovY, float aspect, float near, float far)
	:Component(),
    m_CameraType(CameraType::kPersp), m_Transform(1.0f), m_FovY(fovY), m_Aspect(aspect), m_Near(near), m_Far(far), m_CamUniformBuffer(0), m_FreeMoveSpeed(10.0f),
	m_Left(0.0f), m_Right(0.0f), m_Top(0.0f), m_Bottom(0.0f)
{
   m_ProjMtx = glm::perspective(fovY, aspect, near, far);   
   
   

   //create camera unifrom buffer
   glGenBuffers(1, &m_CamUniformBuffer);
   glBindBuffer(GL_UNIFORM_BUFFER, m_CamUniformBuffer);
   glBufferData(GL_UNIFORM_BUFFER, GetUBufferSize(), nullptr, GL_STATIC_DRAW);
   glBindBuffer(GL_UNIFORM_BUFFER, 0);

}

Camera::Camera(float left, float right, float bottom, float top, float near, float far)
	:Component(),
	m_CameraType(CameraType::kOrtho), m_Transform(1.0f), m_FovY(0.0f), m_Aspect(0.0f), m_Near(near), m_Far(far), m_CamUniformBuffer(0), m_FreeMoveSpeed(10.0f),
	m_Left(left), m_Right(right), m_Top(top), m_Bottom(bottom)
{
	m_ProjMtx = glm::ortho(left, right, bottom, top, near, far);

	//create camera unifrom buffer
	glGenBuffers(1, &m_CamUniformBuffer);
	glBindBuffer(GL_UNIFORM_BUFFER, m_CamUniformBuffer);
	glBufferData(GL_UNIFORM_BUFFER, GetUBufferSize(), nullptr, GL_STATIC_DRAW);
	glBindBuffer(GL_UNIFORM_BUFFER, 0);
}

Camera::~Camera()
{
	if (m_CamUniformBuffer)
	{
		glDeleteBuffers(1, &m_CamUniformBuffer);
		m_CamUniformBuffer = 0;
	}
}

void Camera::Start()
{
    
}

void Camera::Update(float dt)
{
    if (World::GetInst().GetMainCamera() != this)
        return;

	if (World::GetInst().IsKeyDown(GLFW_KEY_W))
	{
        m_Transform = glm::translate(m_Transform, vec3(0.0f, 0.0f, m_FreeMoveSpeed * -dt));
	}

	if (World::GetInst().IsKeyDown(GLFW_KEY_S))
	{
        m_Transform = glm::translate(m_Transform, vec3(0.0f, 0.0f, m_FreeMoveSpeed * dt));
	}

	if (World::GetInst().IsKeyDown(GLFW_KEY_A))
	{
        m_Transform = glm::translate(m_Transform, vec3(m_FreeMoveSpeed * -dt, 0.0f, 0.0f));
	}

	if (World::GetInst().IsKeyDown(GLFW_KEY_D))
	{
        m_Transform = glm::translate(m_Transform, vec3(m_FreeMoveSpeed * dt, 0.0f, 0.0f));
	}

	if (World::GetInst().IsKeyDown(GLFW_KEY_Q))
	{
        m_Transform = glm::translate(m_Transform, vec3(0.0f, m_FreeMoveSpeed * dt, 0.0f));
	}

	if (World::GetInst().IsKeyDown(GLFW_KEY_E))
	{
        m_Transform = glm::translate(m_Transform, vec3(0.0f, m_FreeMoveSpeed * -dt, 0.0f));
	}
}

void Camera::UpdateCamMtx(UniformBufferBinding binding) const
{
	glBindBufferRange(GL_UNIFORM_BUFFER, (uint8_t)binding, m_CamUniformBuffer, 0, GetUBufferSize());
	glBindBuffer(GL_UNIFORM_BUFFER, m_CamUniformBuffer);
    mat4x4 view_mtx(1.0f);
    view_mtx[0][0] = m_Transform[0][0];  view_mtx[0][1] = m_Transform[1][0];  view_mtx[0][2] = m_Transform[2][0];
    view_mtx[1][0] = m_Transform[0][1];  view_mtx[1][1] = m_Transform[1][1];  view_mtx[1][2] = m_Transform[2][1];
    view_mtx[2][0] = m_Transform[0][2];  view_mtx[2][1] = m_Transform[1][2];  view_mtx[2][2] = m_Transform[2][2];
    view_mtx[3][0] = -glm::dot(m_Transform[0], m_Transform[3]);
    view_mtx[3][1] = -glm::dot(m_Transform[1], m_Transform[3]);
    view_mtx[3][2] = -glm::dot(m_Transform[2], m_Transform[3]);

	glBufferSubData(GL_UNIFORM_BUFFER, 0, sizeof(glm::mat4), glm::value_ptr(view_mtx));
	glBufferSubData(GL_UNIFORM_BUFFER, sizeof(glm::mat4), sizeof(glm::mat4), glm::value_ptr(m_ProjMtx));
	glBufferSubData(GL_UNIFORM_BUFFER, 2 * sizeof(glm::mat4), sizeof(glm::vec4), glm::value_ptr(m_Transform[3]));
	glBindBuffer(GL_UNIFORM_BUFFER, 0);
}

void Camera::SetTransform(const mat4x4 & transform)
{
    m_Transform = transform;
}

glm::mat4x4 Camera::GetTransform() const
{
    return m_Transform;
}

glm::vec3 Camera::GetRight() const
{
    return m_Transform[0];
}

glm::vec3 Camera::GetUp() const
{
    return m_Transform[1];
}

glm::vec3 Camera::GetForward() const
{
    return m_Transform[2];
}

glm::vec3 Camera::GetPos() const
{
	return m_Transform[3];
}

glm::mat4x4 Camera::GetViewMtx() const
{
	mat4x4 view_mtx(1.0f);
	view_mtx[0][0] = m_Transform[0][0];  view_mtx[0][1] = m_Transform[1][0];  view_mtx[0][2] = m_Transform[2][0];
	view_mtx[1][0] = m_Transform[0][1];  view_mtx[1][1] = m_Transform[1][1];  view_mtx[1][2] = m_Transform[2][1];
	view_mtx[2][0] = m_Transform[0][2];  view_mtx[2][1] = m_Transform[1][2];  view_mtx[2][2] = m_Transform[2][2];
	view_mtx[3][0] = -glm::dot(m_Transform[0], m_Transform[3]);
	view_mtx[3][1] = -glm::dot(m_Transform[1], m_Transform[3]);
	view_mtx[3][2] = -glm::dot(m_Transform[2], m_Transform[3]);

	return view_mtx;
}

glm::mat4x4 Camera::GetProjMtx() const
{
	return m_ProjMtx;
}

void Camera::SetFreeMoveSpeed(float speed)
{
    m_FreeMoveSpeed = speed;
}

void Camera::MoveTo(const vec3& pos)
{
    m_Transform[3] = glm::vec4(pos, 1.0f);
}

void Camera::LookAtDir(const vec3& dir, const vec3 & up)
{
	const vec4 r = vec4(normalize(cross(up, dir)), 0.0f);
    const vec4 u = vec4(normalize(cross(dir, vec3(r))), 0.0f);
    const vec4 f = vec4(-normalize(dir), 0.0f);
    m_Transform[0] = r;
    m_Transform[1] = u;
    m_Transform[2] = f;
}

void Camera::Rotate(const vec3 & axis, float angle_rad)
{
    m_Transform = glm::rotate(m_Transform, angle_rad, axis);
}
