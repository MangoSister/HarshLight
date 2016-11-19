#include "Model.h"
#include <utility>
#include <cassert>
#include <cstdio>
#include <glm/gtc/matrix_transform.hpp>

Model::Model(Primitive primitive)
{
#ifdef _DEBUG
	assert(primitive < Primitive::kCount);
#endif

	m_Path = "";
	std::vector<glm::vec3> pos
	{
		{0.5f, 0.5f, 0.0f},  // Top Right
		{0.5f, -0.5f, 0.0f},  // Bottom Right
		{-0.5f, -0.5f, 0.0f},  // Bottom Left
		{-0.5f,  0.5f, 0.0f},   // Top Left 
	};

	std::vector<uint32_t> indices
	{
		0, 1, 3,   // First Triangle
		1, 2, 3    // Second Triangle
	};
	std::vector<glm::vec3> normals
	{
		{ 0.0f, 0.0f, -1.0f },  // Top Right
		{ 0.0f, 0.0f, -1.0f },  // Bottom Right
		{ 0.0f, 0.0f, -1.0f },  // Bottom Left
		{ 0.0f, 0.0f, -1.0f },   // Top Left 
	};

	std::vector<glm::vec2> uvs
	{
		{ 1.0f, 0.0f },  // Top Right
		{ 1.0f, 1.0f },  // Bottom Right
		{ 0.0f, 1.0f },  // Bottom Left
		{ 0.0f, 0.0f },  // Top Left 
	};

	Mesh* mesh = new Mesh(std::move(pos), std::move(indices), std::move(normals), std::move(uvs));
	m_Meshes.push_back(mesh);
}

Model::Model(const char* path)
{
#ifdef _DEBUG
	assert(path != nullptr);
#endif
	m_Path = path;
	LoadModel(path);
}

Model::~Model()
{
	for (Mesh*& mesh : m_Meshes)
	{
		if (mesh)
		{
			delete mesh;
			mesh = nullptr;
		}
	}
}

void Model::Render(const Material* material) const
{
    for (Mesh* mesh : m_Meshes)
        mesh->Render(material);
}

void Model::LoadModel(const char* path)
{
	Assimp::Importer import;
	const aiScene* scene = import.ReadFile(path, aiProcess_Triangulate | aiProcess_FlipUVs);

	if (!scene || scene->mFlags == AI_SCENE_FLAGS_INCOMPLETE || !scene->mRootNode)
	{
		fprintf(stderr, "assimp error: %s\n", import.GetErrorString());
		return;
	}

	LoadNode(scene->mRootNode, scene);
}

void Model::LoadNode(const aiNode* node, const aiScene* scene)
{
	// Process all the node's meshes (if any)
	for (uint32_t i = 0; i < node->mNumMeshes; i++)
	{
		aiMesh* mesh = scene->mMeshes[node->mMeshes[i]];
		m_Meshes.push_back(new Mesh(mesh));
	}
	// Then do the same for each of its children
	for (GLuint i = 0; i < node->mNumChildren; i++)
		LoadNode(node->mChildren[i], scene);
}
