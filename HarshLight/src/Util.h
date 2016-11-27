#pragma once
#include "GL/glew.h"
#include <GLFW/glfw3.h>

#define BINDING_POINT_VOXEL_IMG 1

enum class UniformBufferBinding : uint8_t
{
    kMainCam = 0,
    kVoxelSpaceReconstruct = 1,
};

void APIENTRY glDebugOutput(GLenum source,
	GLenum type,
	GLuint id,
	GLenum severity,
	GLsizei length,
	const GLchar* message,
	const GLvoid* userParam);

void DrawTestTriangle();

void DrawPixelGrid(uint32_t mWidth, uint32_t mHeight);