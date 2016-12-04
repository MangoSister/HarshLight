#pragma once

#include "GL/glew.h"

class ComputeShaderProgram
{
public:
    void AddShader(const char* path);
    void LinkProgram();
    GLuint GetProgram() const;

    void Use() const;

    ComputeShaderProgram();
    ~ComputeShaderProgram();

    ComputeShaderProgram(const ComputeShaderProgram& other) = delete;

private:

    GLuint m_ShaderProgram;
    GLuint m_CompShader;
};