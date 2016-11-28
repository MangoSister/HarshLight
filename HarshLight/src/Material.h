#pragma once

#include "Textur2d.h"
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
	void AddTexture(GLuint tex2d, const char* semantic);
	void AddTexture(const Texture3dCompute* tex3d, const char* semantic, TexUsage usage, GLuint binding);
    void DeleteTexture(const char* semantic);
    void DeleteAllTextures();
	void SetShader(ShaderProgram* shader);
	const ShaderProgram* GetShader() const;

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
		Texture3dSlot(GLuint tex3d, const char* semantic, TexUsage usage, GLuint binding) :
			m_Tex3dObj(tex3d), m_Semantic(semantic), m_Usage(usage), m_BindingPoint(binding) {}
	};

    std::vector<Texture2dSlot> m_Textures2d;
	std::vector<Texture3dSlot> m_Textures3d;
};