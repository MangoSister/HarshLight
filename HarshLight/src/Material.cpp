#include "Material.h"
#include <string>
#include <cstdio>
#include <cstdlib>
#include <cassert>
#include <GL/glew.h>
#include <GLFW/glfw3.h>

Material::Material()
    : m_Shader(nullptr){ }

Material::~Material()
{
}

void Material::AddTexture(const Texture2d* tex2d, const char* semantic)
{
#ifdef _DEBUG
    assert(tex2d!= nullptr && semantic != nullptr);
#endif

    m_Textures2d.push_back(Texture2dSlot(tex2d->GetTexObj(), semantic));
}

void Material::AddTexture(GLuint tex2d, const char * semantic)
{
#ifdef _DEBUG
	assert(tex2d != 0 && semantic != nullptr);
#endif
	m_Textures2d.push_back(Texture2dSlot(tex2d, semantic));
}

void Material::AddTexture(const Texture3dCompute * tex3d, const char * semantic, TexUsage usage, GLuint binding)
{
#ifdef _DEBUG
	assert(tex3d != nullptr && semantic != nullptr);
#endif
	m_Textures3d.push_back(Texture3dSlot(tex3d->GetTexObj(), semantic, usage, binding));
}

void Material::DeleteTexture(const char * semantic)
{
    for (auto iter = m_Textures2d.begin(); iter != m_Textures2d.end();)
    {
        if (strcmp(semantic, iter->m_Semantic) == 0)
            iter = m_Textures2d.erase(iter);
        else ++iter;
    }

    for (auto iter = m_Textures3d.begin(); iter != m_Textures3d.end();)
    {
        if (strcmp(semantic, iter->m_Semantic) == 0)
            iter = m_Textures3d.erase(iter);
        else ++iter;
    }
}

void Material::DeleteAllTextures()
{
    m_Textures2d.clear();
    m_Textures3d.clear();
}

void Material::SetShader(ShaderProgram* shader)
{
#ifdef _DEBUG
	assert(shader);
#endif
	m_Shader = shader;
}

const ShaderProgram * Material::GetShader() const
{
	return m_Shader;
}

void Material::SetFloatParam(const char * semantic, float param)
{
    GLuint loc = glGetUniformLocation(m_Shader->GetProgram(), semantic);
    if (loc != -1)
        glProgramUniform1f(m_Shader->GetProgram(), loc, param);
}

void Material::SetVec2Param(const char * semantic, const glm::vec2 & param)
{
    GLuint loc = glGetUniformLocation(m_Shader->GetProgram(), semantic);
    if (loc != -1)
        glProgramUniform2f(m_Shader->GetProgram(), loc, param.x, param.y);
}

void Material::SetVec3Param(const char * semantic, const glm::vec3 & param)
{
    GLuint loc = glGetUniformLocation(m_Shader->GetProgram(), semantic);
    if (loc != -1)
        glProgramUniform3f(m_Shader->GetProgram(), loc, param.x, param.y, param.z);
}

void Material::SetVec4Param(const char * semantic, const glm::vec4 & param)
{
    GLuint loc = glGetUniformLocation(m_Shader->GetProgram(), semantic);
    if (loc != -1)
        glProgramUniform4f(m_Shader->GetProgram(), loc, param.x, param.y, param.z, param.w);
}

void Material::Use() const
{
#ifdef _DEBUG
    assert(m_Shader);
#endif
	m_Shader->Use();
    const uint32_t SAMPLER2D_MAX_NUM = 8;
	uint32_t tex2d_num = (uint32_t)m_Textures2d.size();
	if (tex2d_num > SAMPLER2D_MAX_NUM)
	{
		fprintf(stderr, "material tex2d number exceeds limit (%u > %u)\n", tex2d_num, SAMPLER2D_MAX_NUM);
		tex2d_num = SAMPLER2D_MAX_NUM;
	}
    for (uint32_t i = 0; i < tex2d_num; i++)
    {
        glActiveTexture(GL_TEXTURE0 + i);
        Texture2dSlot tex_slot = m_Textures2d[i];
        glBindTexture(GL_TEXTURE_2D, tex_slot.m_Tex2dObj);
        GLint loc = glGetUniformLocation(m_Shader->GetProgram(), tex_slot.m_Semantic);
        if (loc == -1)
        {
            //fprintf(stderr, "WARNING: cannot find tex2d %s\n", tex_slot.m_Semantic);
            continue;
        }
        glUniform1i(loc, i);
    }

	const uint32_t IMG3D_MAX_NUM = 4;
	uint32_t tex3d_num = (uint32_t)m_Textures3d.size();
	if (tex3d_num > IMG3D_MAX_NUM)
	{
		fprintf(stderr, "material tex3d number exceeds limit (%u > %u)\n", tex3d_num, IMG3D_MAX_NUM);
		tex3d_num = IMG3D_MAX_NUM;
	}
	for (uint32_t i = 0; i < tex3d_num; i++)
	{
		Texture3dSlot tex_slot = m_Textures3d[i];
		GLint loc = glGetUniformLocation(m_Shader->GetProgram(), tex_slot.m_Semantic);
		if (loc == -1)
		{
			//fprintf(stderr, "WARNING: cannot find tex3d variable %s\n", tex_slot.m_Semantic);
			continue;
		}

		if (tex_slot.m_Usage == TexUsage::kRegularTexture)
		{
			glActiveTexture(GL_TEXTURE0 + tex2d_num + i);
			glBindTexture(GL_TEXTURE_3D, tex_slot.m_Tex3dObj);
			glUniform1i(loc, tex2d_num + i);
		}
		else
		{
			if (tex_slot.m_Usage == TexUsage::kImageReadOnly)
				glBindImageTexture(tex_slot.m_BindingPoint, tex_slot.m_Tex3dObj, 0, GL_TRUE, 0, GL_READ_ONLY, GL_RGBA8);
			else if (tex_slot.m_Usage == TexUsage::kImageWriteOnly)
				glBindImageTexture(tex_slot.m_BindingPoint, tex_slot.m_Tex3dObj, 0, GL_TRUE, 0, GL_WRITE_ONLY, GL_RGBA8);
			else if (tex_slot.m_Usage == TexUsage::kImageReadWrite)
				glBindImageTexture(tex_slot.m_BindingPoint, tex_slot.m_Tex3dObj, 0, GL_TRUE, 0, GL_READ_WRITE, GL_RGBA8);
		}
	}
}