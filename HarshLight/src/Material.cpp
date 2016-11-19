#include "Material.h"
#include <cstdio>
#include <cstdlib>
#include <cassert>

Material::Material()
    : m_ShaderProgram(0), m_ShaderTypeMask(0), m_VertShader(0), m_GeomShader(0), m_FragShader(0) { }

Material::~Material()
{
    if (m_ShaderTypeMask & VERTEX && m_VertShader)
    {
        glDeleteShader(m_VertShader);
        m_VertShader = 0;
    }
    if (m_ShaderTypeMask & GEOMETRY && m_GeomShader)
    {
        glDeleteShader(m_GeomShader);
        m_GeomShader = 0;
    }
    if (m_ShaderTypeMask & FRAGMENT && m_FragShader)
    {
        glDeleteShader(m_FragShader);
        m_FragShader = 0;
    }

    if (m_ShaderProgram)
        glDeleteProgram(m_ShaderProgram);
}

void Material::AddVertShader(const char* path)
{
    m_ShaderTypeMask |= VERTEX;
    FILE* file = std::fopen(path, "r");
    if (!file)
    {
        fprintf(stderr, "ERROR: fail to open vertex shader file: %s\n", path);
        return;
    }
    std::fseek(file, 0, SEEK_END);
    long fsize = ftell(file);
    std::rewind(file);

    GLchar* shader_code = (GLchar*)calloc(fsize + 1, sizeof(GLchar));
    std::fread(shader_code, 1, fsize, file);
    if(std::ferror(file))
    {
        fprintf(stderr, "ERROR: fail to read vertex shader file: %s\n", path);
        std::fclose(file);
        free(shader_code);
        return;
    }

    std::fclose(file);

    m_VertShader = glCreateShader(GL_VERTEX_SHADER);
    glShaderSource(m_VertShader, 1, &shader_code, nullptr);
    glCompileShader(m_VertShader);
    GLint succ;
    glGetShaderiv(m_VertShader, GL_COMPILE_STATUS, &succ);
    if (!succ)
    {
        GLchar log[512];
        glGetShaderInfoLog(m_VertShader, 512, nullptr, log);
        fprintf(stderr, "ERROR: vertex shader compiling error: %s\n", log);
    }

    free(shader_code);
}

void Material::AddGeomShader(const char* path)
{
    m_ShaderTypeMask |= GEOMETRY;
    FILE* file = std::fopen(path, "r");
    if (!file)
    {
        fprintf(stderr, "ERROR: fail to open geometry shader file: %s\n", path);
        return;
    }
    std::fseek(file, 0, SEEK_END);
    long fsize = ftell(file);
    std::rewind(file);

    GLchar* shader_code = (GLchar*)calloc(fsize + 1, sizeof(GLchar));
    std::fread(shader_code, 1, fsize, file);
    if (std::ferror(file))
    {
        fprintf(stderr, "ERROR: fail to read geometry shader file: %s\n", path);
        std::fclose(file);
        free(shader_code);
        return;
    }

    std::fclose(file);

    m_GeomShader = glCreateShader(GL_GEOMETRY_SHADER);
    glShaderSource(m_GeomShader, 1, &shader_code, nullptr);
    glCompileShader(m_GeomShader);
    GLint succ;
    glGetShaderiv(m_GeomShader, GL_COMPILE_STATUS, &succ);
    if (!succ)
    {
        GLchar log[512];
        glGetShaderInfoLog(m_GeomShader, 512, nullptr, log);
        fprintf(stderr, "ERROR: geometry shader compiling error: %s\n", log);
    }

    free(shader_code);
}

void Material::AddFragShader(const char* path)
{
    m_ShaderTypeMask |= FRAGMENT;
    FILE* file = std::fopen(path, "r");
    if (!file)
    {
        fprintf(stderr, "ERROR: fail to open fragment shader file: %s\n", path);
        return;
    }
    std::fseek(file, 0, SEEK_END);
    long fsize = ftell(file);
    std::rewind(file);  //same as rewind(f);

    GLchar* shader_code = (GLchar*)calloc(fsize + 1, sizeof(GLchar));
    std::fread(shader_code, 1, fsize, file);
    if (std::ferror(file))
    {
        fprintf(stderr, "ERROR: fail to read fragment shader file: %s\n", path);
        std::fclose(file);
        free(shader_code);
        return;
    }

    std::fclose(file);

    m_FragShader = glCreateShader(GL_FRAGMENT_SHADER);
    glShaderSource(m_FragShader, 1, &shader_code, nullptr);
    glCompileShader(m_FragShader);
    GLint succ;
    glGetShaderiv(m_FragShader, GL_COMPILE_STATUS, &succ);
    if (!succ)
    {
        GLchar log[512];
        glGetShaderInfoLog(m_FragShader, 512, nullptr, log);
        fprintf(stderr, "ERROR: fragment shader compiling error: %s\n", log);
    }

    free(shader_code);
}

void Material::LinkProgram()
{
#ifdef _DEBUG
    assert(!m_ShaderProgram); //no relink
    assert(m_ShaderTypeMask); //at least one shader?
#endif

    m_ShaderProgram = glCreateProgram();
    if (m_ShaderTypeMask & VERTEX)
        glAttachShader(m_ShaderProgram, m_VertShader);
    if (m_ShaderTypeMask & GEOMETRY)
        glAttachShader(m_ShaderProgram, m_GeomShader);
    if (m_ShaderTypeMask & FRAGMENT)
        glAttachShader(m_ShaderProgram, m_FragShader);
    glLinkProgram(m_ShaderProgram);
    GLint succ;
    glGetProgramiv(m_ShaderProgram, GL_LINK_STATUS, &succ);
    if (!succ)
    {
        char log[512];
        glGetProgramInfoLog(m_ShaderProgram, 512, nullptr, log);
        fprintf(stderr, "ERROR: shader linking error %s\n", log);
        return;
    }

    if (m_ShaderTypeMask & VERTEX)
    {
        glDeleteShader(m_VertShader);
        m_VertShader = 0;
    }
    if (m_ShaderTypeMask & GEOMETRY)
    {
        glDeleteShader(m_GeomShader);
        m_GeomShader = 0;
    }
    if (m_ShaderTypeMask & FRAGMENT)
    {
        glDeleteShader(m_FragShader);
        m_FragShader = 0;
    }
}

GLuint Material::GetProgram() const
{
	return m_ShaderProgram;
}

void Material::Use() const
{
#ifdef _DEBUG
    assert(m_ShaderProgram);
#endif
    glUseProgram(m_ShaderProgram);
}
