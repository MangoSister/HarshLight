#include "ComputeShaderProgram.h"
#include <string>
#include <cstdio>
#include <cassert>
#include <GL/glew.h>
#include <GLFW/glfw3.h>

ComputeShaderProgram::ComputeShaderProgram()
    :m_ShaderProgram(0), m_CompShader(0) { }

ComputeShaderProgram::~ComputeShaderProgram()
{
    if (m_CompShader)
    {
        glDeleteShader(m_CompShader);
        m_CompShader = 0;
    }

    if (m_ShaderProgram)
    {
        glDeleteProgram(m_ShaderProgram);
        m_ShaderProgram = 0;
    }
}

void ComputeShaderProgram::AddShader(const char* path)
{
    FILE* file = std::fopen(path, "r");
    if (!file)
    {
        fprintf(stderr, "ERROR: fail to open compute shader file: %s\n", path);
        return;
    }

    std::fseek(file, 0, SEEK_END);
    long fsize = ftell(file);
    std::rewind(file);

    GLchar* shader_code = (GLchar*)calloc(fsize + 1, sizeof(GLchar));
    std::fread(shader_code, 1, fsize, file);
    if (std::ferror(file))
    {
        fprintf(stderr, "ERROR: fail to read compute shader file: %s\n", path);
        std::fclose(file);
        free(shader_code);
        return;
    }

    std::fclose(file);

    m_CompShader = glCreateShader(GL_COMPUTE_SHADER);
    glShaderSource(m_CompShader, 1, &shader_code, nullptr);
    glCompileShader(m_CompShader);
    GLint succ;
    glGetShaderiv(m_CompShader, GL_COMPILE_STATUS, &succ);
    if (!succ)
    {
        GLchar log[512];
        glGetShaderInfoLog(m_CompShader, 512, nullptr, log);
        fprintf(stderr, "ERROR: compute shader compiling error: %s\n", log);
    }

    free(shader_code);
}

void ComputeShaderProgram::LinkProgram()
{
#ifdef _DEBUG
    assert(!m_ShaderProgram); //no relink
#endif

    m_ShaderProgram = glCreateProgram();
    glAttachShader(m_ShaderProgram, m_CompShader);
    glLinkProgram(m_ShaderProgram);
    GLint succ;
    glGetProgramiv(m_ShaderProgram, GL_LINK_STATUS, &succ);
    if (!succ)
    {
        char log[512];
        glGetProgramInfoLog(m_ShaderProgram, 512, nullptr, log);
        fprintf(stderr, "ERROR: compute shader linking error %s\n", log);
        return;
    }

    glDeleteShader(m_CompShader);
    m_CompShader = 0;
}

GLuint ComputeShaderProgram::GetProgram() const
{
    return m_ShaderProgram;
}

void ComputeShaderProgram::Use() const
{
    glUseProgram(m_ShaderProgram);
}
