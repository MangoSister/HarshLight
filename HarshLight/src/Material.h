#pragma once

#include "Textur2d.h"
#include "GL/glew.h"
#include <cstdint>
#include <vector>

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

    void AddTexture(const Texture2d* tex2d, const char* semantic);

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

    struct Texture2dSlot
    {
    public:
        const Texture2d* m_Tex2d;
        const char* m_Semantic;
        Texture2dSlot(const Texture2d* tex2d, const char* semantic) :
            m_Tex2d(tex2d), m_Semantic(semantic) {}
    };

    std::vector<Texture2dSlot> m_Textures;
};