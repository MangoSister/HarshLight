#pragma once

#include <GL/glew.h>
#include <glfw/glfw3.h>

class Texture2d
{
public:
    explicit Texture2d(const char* path);
	Texture2d(const Texture2d& other) = delete;
    ~Texture2d();
    
    GLuint GetTexObj() const;

private:
    const char* m_RawPath;
    GLuint m_TexObject;
};

class Texture3dCompute
{
public:
	explicit Texture3dCompute(uint32_t dim_x, uint32_t dim_y, uint32_t dim_z);
	Texture3dCompute(const Texture3dCompute& other) = delete;
	~Texture3dCompute();
	void CleanContent();
	GLuint GetTexObj() const;
private:

	uint32_t m_DimX;
	uint32_t m_DimY;
	uint32_t m_DimZ;
	GLuint m_Format;

	GLuint m_TexObject;
	GLuint m_UtilFBO;
};