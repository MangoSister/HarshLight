#include "Model.h"
#include <cassert>
#include <cstdio>

Model::Model(const char* path)
{
	assert(path != nullptr);
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
