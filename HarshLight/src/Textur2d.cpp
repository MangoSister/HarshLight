#include "Textur2d.h"
#include "SOIL/SOIL.h"
#include <cassert>
#include <cstdio>

Texture2d::Texture2d(const char * path)
{
    m_TexObject = 0;
    m_RawPath = nullptr;

#ifdef _DEBUG
    assert(path != nullptr);
#endif
    int32_t width, height, channel;
    uint8_t* image = SOIL_load_image(path, &width, &height, &channel, SOIL_LOAD_RGB);
    if (!image)
    {
        fprintf(stderr, "fail to load texture: %s\n", path);
        return;
    }
    
    m_RawPath = path;
    glGenTextures(1, &m_TexObject);
#ifdef _DEBUG
    assert(m_TexObject);
#endif
    glBindTexture(GL_TEXTURE_2D, m_TexObject);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, width, height, 0, GL_RGB, GL_UNSIGNED_BYTE, image);
    glGenerateMipmap(GL_TEXTURE_2D);
    SOIL_free_image_data(image);
    glBindTexture(GL_TEXTURE_2D, 0);
}

Texture2d::~Texture2d()
{
    if (m_TexObject)
    {
        glDeleteTextures(1, &m_TexObject);
        m_TexObject = 0;
    }
}

GLuint Texture2d::GetTexObj() const
{
    return m_TexObject;
}
