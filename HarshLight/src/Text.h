#pragma once

#include <map>
#include <GL/glew.h>
#include <GLFW/glfw3.h>
#include <glm/glm.hpp>
#include <ft2build.h>
#include FT_FREETYPE_H
#include "ShaderProgram.h"

struct Character 
{
    GLuint m_TexObj;   // ID handle of the glyph texture
    glm::ivec2 m_Size;    // Size of glyph
    glm::ivec2 m_Bearing;  // Offset from baseline to left/top of glyph
    GLuint m_Advance;    // Horizontal offset to advance to next glyph
};

class TextManager
{
public:
    ~TextManager();
    void Init();
    void RenderText(const std::string& text, GLfloat x, GLfloat y, GLfloat scale, glm::vec3 color);

private:
    std::map<GLchar, Character> m_Characters;
	GLuint m_VAO;
	GLuint m_VBO;
    ShaderProgram m_FontShader;
    glm::mat4x4 m_ProjMtx;
};