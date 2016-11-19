#pragma once

#include <vector>
#include <assimp/Importer.hpp>
#include <assimp/scene.h>
#include <assimp/postprocess.h>
#include "glm/glm.hpp"
#include "GL/glew.h"
#include "Material.h"

class Mesh
{
public:
	explicit Mesh(std::vector<glm::vec3>&& pos, std::vector<uint32_t>&& indices,
		std::vector<glm::vec3>&& normals, std::vector<glm::vec2>&& uvs);

	explicit Mesh(const aiMesh* aiMesh);
	~Mesh();
	void Render(const Material* shader) const;

private:
	void CreateBuffers();
	
	std::vector<glm::vec3> m_Positions;
	std::vector<uint32_t> m_Indices;
	std::vector<glm::vec3> m_Normals;
	std::vector<glm::vec2> m_Uvs;
	
	GLuint m_VAO;
	GLuint m_PosVBO;
	GLuint m_NrmVBO;
	GLuint m_UvVBO;
	GLuint m_EBO;

};