#include <glm/gtc/matrix_transform.hpp>
#include "Camera.h"

Camera::Camera(float fovY, float aspect, float near, float far)
    :m_ViewMtx(1.0f), m_FovY(fovY), m_Aspect(aspect), m_Near(near), m_Far(far)
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
   
}
