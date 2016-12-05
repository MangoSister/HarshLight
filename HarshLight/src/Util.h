#pragma once
#include "GL/glew.h"
#include <GLFW/glfw3.h>
#include <cuda_runtime.h>
#include <cstdlib>
#include <cstdio>
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>

#define NOMINMAX

#define BINDING_POINT_START_VOXEL_IMG 1

enum class UniformBufferBinding : uint8_t
{
	kMainCam = 0,
	kVoxelSpaceReconstruct = 1,
	kLight = 2,
    kPointLightCapture = 3,
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



#ifdef _DEBUG
#define cudaCheckError(ans) { cudaAssert((ans), __FILE__, __LINE__); }
inline void cudaAssert(cudaError_t code, const char *file, int line, bool abort = true)
{
	if (code != cudaSuccess)
	{
		fprintf(stderr, "CUDA Error: %s at %s:%d\n",
			cudaGetErrorString(code), file, line);
		if (abort) exit(code);
	}
}
#else
#define cudaCheckError(ans) ans
#endif

#ifdef _DEBUG
#define HANDLE_KERNEL_ERROR_SYNC \
  cudaCheckError(cudaPeekAtLastError()); \
  cudaCheckError(cudaDeviceSynchronize());
#else 
#define HANDLE_KERNEL_ERROR_SYNC
#endif

glm::mat4x4 LookAtDir(const glm::vec3& dir, const glm::vec3& up);
glm::mat4x4 ViewMtxFromTransform(const glm::mat4x4& transform);