#pragma once

#include "GL/glew.h"
#include <cstdint>

class Material
{
public:
    
    typedef uint8_t ShaderTypeMask;

    enum
    {
        VERTEX = 0x01,
        GEOMETRY = 0x02,
        FRAGMENT = 0x04,
    };

    explicit Material();
    ~Material();

    void AddVertShader(const char* path);
    void AddGeomShader(const char* path);
    void AddFragShader(const char* path);
    void LinkProgram();
	GLuint GetProgram() const;

    void Use() const;
private:
    GLuint m_ShaderProgram;

    ShaderTypeMask m_ShaderTypeMask;
    GLuint m_VertShader;
    GLuint m_GeomShader;
    GLuint m_FragShader;
};