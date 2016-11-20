#pragma once

#include <GL/glew.h>
#include <glfw/glfw3.h>

class Texture2d
{
public:
    explicit Texture2d(const char* path);
    ~Texture2d();
    
    GLuint GetTexObj() const;

private:
    const char* m_RawPath;
    GLuint m_TexObject;
};