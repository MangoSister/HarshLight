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

    void SetMaterialIndex(uint32_t idx);
    uint32_t GetMaterialIndex() const;

    inline const glm::vec3& GetBBoxMin() const
    { return m_BBoxMin; }

    inline const glm::vec3& GetBBoxMax() const
    { return m_BBoxMax; }

private:
	void CreateBuffers();
	
	std::vector<glm::vec3> m_Positions;
	std::vector<uint32_t> m_Indices;
	std::vector<glm::vec3> m_Normals;
	std::vector<glm::vec2> m_Uvs;
    std::vector<glm::vec3> m_Tangents;

	GLuint m_VAO;
	GLuint m_PosVBO;
	GLuint m_NrmVBO;
	GLuint m_UvVBO;
    GLuint m_TanVBO;
	GLuint m_EBO;

    uint32_t m_MaterialIndex;

    glm::vec3 m_BBoxMin;
    glm::vec3 m_BBoxMax;
};