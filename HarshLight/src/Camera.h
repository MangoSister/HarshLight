#include <glm/vec3.hpp>
#include <glm/mat4x3.hpp>
#include <glm/mat4x4.hpp>
#include "Component.h"
#pragma once

using namespace glm;

class Camera : Component
{
public:
    explicit Camera(float fovY, float aspect, float near, float far);
    ~Camera();

    void LookAt(const vec3& target, const vec3& up);

    void Start() override;
    void Update(float dt) override;

private:

    mat4x4 m_ViewMtx;
    
    mat4x4 m_ProjMtx;
    float m_FovY;
    float m_Aspect;
    float m_Near;
    float m_Far;

};