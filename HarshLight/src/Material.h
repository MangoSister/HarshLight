#pragma once

#include "Texture.h"
#include "ShaderProgram.h"
#include "GL/glew.h"
#include <cstdint>
#include <vector>
#include <glm/glm.hpp>

enum class TexUsage : uint8_t
{
	kRegularTexture = 0,
	kImageReadOnly = 1,
	kImageWriteOnly = 2,
	kImageReadWrite = 3,
};

class Material
{
public:
    
    explicit Material();
    ~Material();

    void AddTexture(const Texture2d* tex2d, const char* semantic);
	void AddTexture2dDirect(GLuint tex2d, const char* semantic);
	void AddTexture(const Texture3dCompute* tex3d, const char* semantic, TexUsage usage, GLuint binding);
    void AddTextureCubeDirect(GLuint tex_cube, const char* semantic);
    void DeleteTexture(const char* semantic);
    void DeleteAllTextures();
	void SetShader(ShaderProgram* shader);
	const ShaderProgram* GetShader() const;

    void SetI32Param(const char* semantic, GLint param);
    void SetFloatParam(const char* semantic, float param);
    void SetVec2Param(const char* semantic, const glm::vec2& param);
    void SetVec3Param(const char* semantic, const glm::vec3& param);
    void SetVec4Param(const char* semantic, const glm::vec4& param);

    void SetMat4x4Param(const char* semantic, const glm::mat4x4& param);

    void Use() const;
   

private:
	
	ShaderProgram* m_Shader;

    struct Texture2dSlot
    {
    public:
		GLuint m_Tex2dObj;
        const char* m_Semantic;
        Texture2dSlot(GLuint tex2d, const char* semantic) :
			m_Tex2dObj(tex2d), m_Semantic(semantic) {}
    };

	struct Texture3dSlot
	{
	public:
		GLuint m_Tex3dObj;
		const char* m_Semantic;
		TexUsage m_Usage;
        GLuint m_BindingPoint;
		GLuint m_InternalFormat;
		Texture3dSlot(GLuint tex3d, const char* semantic, TexUsage usage, GLuint binding, GLuint internal_format) :
			m_Tex3dObj(tex3d), m_Semantic(semantic), m_Usage(usage), m_BindingPoint(binding), m_InternalFormat(internal_format) {}
	};

    struct TextureCubeSlot
    {
    public:
        GLuint m_TexCubeObj;
        const char* m_Semantic;
        TextureCubeSlot(GLuint tex_cube, const char* semantic) :
            m_TexCubeObj(tex_cube), m_Semantic(semantic) {}
    };

    std::vector<Texture2dSlot> m_Textures2d;
	std::vector<Texture3dSlot> m_Textures3d;
    std::vector<TextureCubeSlot> m_TexturesCube;
};