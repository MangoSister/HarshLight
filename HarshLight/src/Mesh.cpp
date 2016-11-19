#include "Mesh.h"

Mesh::Mesh(std::vector<glm::vec3>&& pos, std::vector<uint32_t>&& indices,
	std::vector<glm::vec3>&& normals, std::vector<glm::vec2>&& uvs)
	:m_Positions(pos), m_Indices(indices), m_Normals(normals), m_Uvs(uvs)
{
	CreateBuffers();
}

Mesh::Mesh(const aiMesh* aiMesh)
{
	for (uint32_t i = 0; i < aiMesh->mNumVertices; i++)
	{
		glm::vec3 pos;
		pos.x = aiMesh->mVertices[i].x;
		pos.y = aiMesh->mVertices[i].y;
		pos.z = aiMesh->mVertices[i].z;
		m_Positions.push_back(pos);

		glm::vec3 normal;
		normal.x = aiMesh->mNormals[i].x;
		normal.y = aiMesh->mNormals[i].y;
		normal.z = aiMesh->mNormals[i].z;
		m_Normals.push_back(normal);
	}

	if (aiMesh->GetNumUVChannels())
	{
		for (uint32_t i = 0; i < aiMesh->mNumVertices; i++)
		{
			glm::vec2 uv;
			uv.x = aiMesh->mTextureCoords[0][i].x;
			uv.y = aiMesh->mTextureCoords[0][i].y;
			m_Uvs.push_back(uv);
		}

	}

	for (uint32_t i = 0; i < aiMesh->mNumFaces; i++)
	{
		aiFace face = aiMesh->mFaces[i];
		for (uint32_t j = 0; j < face.mNumIndices; j++)
			m_Indices.push_back(face.mIndices[j]);
	}

	CreateBuffers();
}

void Mesh::CreateBuffers()
{
	glGenVertexArrays(1, &m_VAO);
	glBindVertexArray(m_VAO);

	glGenBuffers(1, &m_PosVBO);
	glBindBuffer(GL_ARRAY_BUFFER, m_PosVBO);
	glBufferData(GL_ARRAY_BUFFER, m_Positions.size() * sizeof(glm::vec3), &m_Positions[0], GL_STATIC_DRAW);
	glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 0, nullptr);
	glEnableVertexAttribArray(0);

	glGenBuffers(1, &m_NrmVBO);
	glBindBuffer(GL_ARRAY_BUFFER, m_NrmVBO);
	glBufferData(GL_ARRAY_BUFFER, m_Normals.size() * sizeof(glm::vec3), &m_Normals[0], GL_STATIC_DRAW);
	glVertexAttribPointer(1, 3, GL_FLOAT, GL_TRUE, 0, nullptr);
	glEnableVertexAttribArray(1);

	glGenBuffers(1, &m_UvVBO);
	glBindBuffer(GL_ARRAY_BUFFER, m_UvVBO);
	glBufferData(GL_ARRAY_BUFFER, m_Uvs.size() * sizeof(glm::vec2), &m_Uvs[0], GL_STATIC_DRAW);
	glVertexAttribPointer(2, 2, GL_FLOAT, GL_FALSE, 0, nullptr);
	glEnableVertexAttribArray(2);

	glGenBuffers(1, &m_EBO);
	glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, m_EBO);
	glBufferData(GL_ELEMENT_ARRAY_BUFFER, m_Indices.size() * sizeof(uint32_t), &m_Indices[0], GL_STATIC_DRAW);

	glBindBuffer(GL_ARRAY_BUFFER, 0);
	glBindVertexArray(0);
}

Mesh::~Mesh()
{
	glDeleteBuffers(1, &m_PosVBO);
	glDeleteBuffers(1, &m_NrmVBO);
	glDeleteBuffers(1, &m_UvVBO);
	glDeleteBuffers(1, &m_EBO);
	glDeleteVertexArrays(1, &m_VAO);
}

void Mesh::Render(const Material* material) const
{
    glBindVertexArray(m_VAO);
    glDrawElements(GL_TRIANGLES, static_cast<GLsizei>(m_Indices.size()), GL_UNSIGNED_INT, 0);
    glBindVertexArray(0);
    
}
