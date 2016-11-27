#pragma once

#include <vector>
#include <assimp/Importer.hpp>
#include <assimp/scene.h>
#include <assimp/postprocess.h>
#include "Mesh.h"
#include "Material.h"

class Model
{
public:
	enum class Primitive : uint8_t
	{
		kQuad = 0,
        kTriangle = 1,
		kCount = 2,
	};

	explicit Model(Primitive primitive);
	explicit Model(const char* path);
	Model(const Model& other) = delete;
	Model& operator=(const Model& other) = delete;

	~Model();
    const char* GetRawPath() const;
    void Render(const glm::mat4x4& transform, const std::vector<Material*>& materials) const;

private:

	void LoadModel(const char* path);
	void LoadNode(const aiNode* node, const aiScene* scene);
    
	std::vector<Mesh*> m_Meshes;
	const char* m_RawPath;

};