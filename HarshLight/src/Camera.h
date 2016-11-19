#include <glm/vec3.hpp>
#include <glm/mat4x3.hpp>
#include <glm/mat4x4.hpp>
#include "Component.h"
#pragma once

using namespace glm;

class Camera : public Component
{
public:
    explicit Camera(float fovY, float aspect, float near, float far);
    ~Camera();

	void MoveTo(const vec3& pos);
	void LookAtDir(const vec3& dir, const vec3& up);
	void Rotate(const vec3& axis, float angle_rad);
    void Start() override;
    void Update(float dt) override;

	void UpdateCamMtx() const;

	glm::vec3 GetRight() const;
	glm::vec3 GetUp() const;
	glm::vec3 GetForward() const;
	glm::vec3 GetPos() const;

    void SetFreeMoveSpeed(float speed);

private:

    mat4x4 m_Transform;
    mat4x4 m_ProjMtx;
    float m_FovY;
    float m_Aspect;
    float m_Near;
    float m_Far;
	float m_FreeMoveSpeed;

	GLuint m_CamUniformBuffer;
};