#include "Model.h"
#include <utility>
#include <cassert>
#include <cstdio>
#include <glm/gtc/type_ptr.hpp>
#include <glm/gtc/matrix_transform.hpp>

Model::Model(Primitive primitive)
{
#ifdef _DEBUG
	assert(primitive < Primitive::kCount);
#endif

	m_RawPath = "";

    switch (primitive)
    {
    case Model::Primitive::kTriangle:
    {
        std::vector<glm::vec3> pos
        {
            {-0.8f, -0.5f, 0.0f}, // Left  
            {0.8f, -0.5f, 0.0f}, // Right 
            {0.0f, 0.5f, 0.0f }  // Top  
        };

        std::vector<uint32_t> indices
        {
            0, 1, 2,   // First Triangle
        };
        std::vector<glm::vec3> normals
        {
            { 0.0f, 0.0f, -1.0f },  // Left  
            { 0.0f, 0.0f, -1.0f },  // Right 
            { 0.0f, 0.0f, -1.0f },  // Top  
        };

        std::vector<glm::vec2> uvs
        {
            { 0.0f, 0.0f },  // Left  
            { 0.0f, 1.0f },  // Right 
            { 1.0f, 0.5f },  // Top  
        };

        Mesh* mesh = new Mesh(std::move(pos), std::move(indices), std::move(normals), std::move(uvs));
        m_Meshes.push_back(mesh);
    }
        break;
    case Model::Primitive::kQuad:
    {
        std::vector<glm::vec3> pos
        {
            { 0.5f, 0.5f, 0.0f },  // Top Right
            { 0.5f, -0.5f, 0.0f },  // Bottom Right
            { -0.5f, -0.5f, 0.0f },  // Bottom Left
            { -0.5f,  0.5f, 0.0f },   // Top Left 
        };

        std::vector<uint32_t> indices
        {
            0, 3, 1,   // First Triangle
            1, 3, 2    // Second Triangle
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
            { 1.0f, 1.0f },  // Top Right
            { 1.0f, 0.0f },  // Bottom Right
            { 0.0f, 0.0f },  // Bottom Left
            { 0.0f, 1.0f },  // Top Left 
        };

        Mesh* mesh = new Mesh(std::move(pos), std::move(indices), std::move(normals), std::move(uvs));
        m_Meshes.push_back(mesh);
    }
        break;
    default:
        break;
    }
}

Model::Model(const char* path)
{
#ifdef _DEBUG
	assert(path != nullptr);
#endif
	m_RawPath = path;
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

const char* Model::GetRawPath() const
{
    return m_RawPath;
}

void Model::Render(const glm::mat4x4& transform, const std::vector<Material*>& materials) const
{
#if _DEBUG
    assert(materials.size() > 0);
#endif

    for (size_t i = 0; i < m_Meshes.size(); i++)
    {
        uint32_t material_idx = m_Meshes[i]->GetMaterialIndex();
        if (material_idx >= materials.size())
        {
            fprintf(stderr, "WARNING: mesh material idx (%u) exceed material array size\n", material_idx);
            material_idx = (uint32_t)materials.size() - 1;
        }
        const Material* material = materials[material_idx];
#if _DEBUG
        assert(material != nullptr);
#endif
        material->Use();
        GLuint model_loc = glGetUniformLocation(material->GetShader()->GetProgram(), "Model");
#if _DEBUG
        if (model_loc == GL_INVALID_VALUE)
            fprintf(stderr, "WARNING: Invalid model mtx shader program location\n");
#endif
        glUniformMatrix4fv(model_loc, 1, GL_FALSE, glm::value_ptr(transform));
        m_Meshes[i]->Render(material);
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
