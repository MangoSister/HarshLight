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

    m_BBoxMin = glm::vec3(FLT_MAX, FLT_MAX, FLT_MAX);
    m_BBoxMax = glm::vec3(FLT_MIN, FLT_MIN, FLT_MIN);

    for (Mesh* mesh : m_Meshes)
    {
        const glm::vec3& mesh_min = mesh->GetBBoxMin();
        const glm::vec3& mesh_max = mesh->GetBBoxMax();

        if (mesh_min.x < m_BBoxMin.x)
            m_BBoxMin.x = mesh_min.x;
        if (mesh_min.y < m_BBoxMin.y)
            m_BBoxMin.y = mesh_min.y;
        if (mesh_min.z < m_BBoxMin.z)
            m_BBoxMin.z = mesh_min.z;

        if (mesh_max.x > m_BBoxMax.x)
            m_BBoxMax.x = mesh_max.x;
        if (mesh_max.y > m_BBoxMax.y)
            m_BBoxMax.y = mesh_max.y;
        if (mesh_max.z > m_BBoxMax.z)
            m_BBoxMax.z = mesh_max.z;
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
        GLint model_loc = glGetUniformLocation(material->GetShader()->GetProgram(), "Model");
#if _DEBUG
        if (model_loc == -1)
            fprintf(stderr, "WARNING: Invalid model mtx shader program location\n");
#endif
        glUniformMatrix4fv(model_loc, 1, GL_FALSE, glm::value_ptr(transform));
        m_Meshes[i]->Render(material);
    }
}

void Model::Render(const glm::mat4x4 & transform, const std::vector<Material*>& materials, const glm::vec3& center, float radius) const
{
#if _DEBUG
    assert(materials.size() > 0);
#endif

    glm::mat4x4 w2o_transform = glm::inverse(transform);

    for (size_t i = 0; i < m_Meshes.size(); i++)
    {
        glm::vec4 obj_pos = glm::vec4(center, 1.0f);
        obj_pos = w2o_transform * obj_pos;
        glm::vec3 best;

        const glm::vec3& min = m_Meshes[i]->GetBBoxMin();
        const glm::vec3& max = m_Meshes[i]->GetBBoxMax();

        best.x = std::max(min.x, std::min(obj_pos.x, max.x));
        best.y = std::max(min.y, std::min(obj_pos.y, max.y));
        best.z = std::max(min.z, std::min(obj_pos.z, max.z));
        glm::vec3 offset = best - glm::vec3(obj_pos);
        float best_dist_sq = glm::dot(offset, offset);
        if (best_dist_sq > radius * radius)
            continue;

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
        GLint model_loc = glGetUniformLocation(material->GetShader()->GetProgram(), "Model");
#if _DEBUG
        if (model_loc == -1)
            fprintf(stderr, "WARNING: Invalid model mtx shader program location\n");
#endif
        glUniformMatrix4fv(model_loc, 1, GL_FALSE, glm::value_ptr(transform));
        m_Meshes[i]->Render(material);
    }
}

void Model::LoadModel(const char* path)
{
	Assimp::Importer import;
	const aiScene* scene = import.ReadFile(path,
		aiProcess_FlipUVs | aiProcess_PreTransformVertices |
		aiProcess_FlipWindingOrder | // seems like models we use are all CW order...
		aiProcess_FindDegenerates |
		aiProcess_OptimizeMeshes |
		aiProcess_OptimizeGraph |
		aiProcess_JoinIdenticalVertices |
		aiProcess_CalcTangentSpace |
		aiProcess_GenSmoothNormals |
		aiProcess_Triangulate |
		aiProcess_FixInfacingNormals |
		aiProcess_FindInvalidData |
		aiProcess_ValidateDataStructure);

	if (!scene || scene->mFlags == AI_SCENE_FLAGS_INCOMPLETE || !scene->mRootNode)
	{
		fprintf(stderr, "assimp error: %s\n", import.GetErrorString());
		return;
	}

	LoadNode(scene->mRootNode, scene);

    m_BBoxMin = glm::vec3(FLT_MAX, FLT_MAX, FLT_MAX);
    m_BBoxMax = glm::vec3(FLT_MIN, FLT_MIN, FLT_MIN);

    for (Mesh* mesh : m_Meshes)
    {
        const glm::vec3& mesh_min = mesh->GetBBoxMin();
        const glm::vec3& mesh_max = mesh->GetBBoxMax();

        if (mesh_min.x < m_BBoxMin.x)
            m_BBoxMin.x = mesh_min.x;
        if (mesh_min.y < m_BBoxMin.y)
            m_BBoxMin.y = mesh_min.y;
        if (mesh_min.z < m_BBoxMin.z)
            m_BBoxMin.z = mesh_min.z;

        if (mesh_max.x > m_BBoxMax.x)
            m_BBoxMax.x = mesh_max.x;
        if (mesh_max.y > m_BBoxMax.y)
            m_BBoxMax.y = mesh_max.y;
        if (mesh_max.z > m_BBoxMax.z)
            m_BBoxMax.z = mesh_max.z;
    }
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
