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
	explicit Model(const char* path);
	Model(const Model& other) = delete;
	Model& operator=(const Model& other) = delete;

	~Model();

    void Render(const Material* shader) const;

private:

	void LoadModel(const char* path);
	void LoadNode(const aiNode* node, const aiScene* scene);

	std::vector<Mesh*> m_Meshes;
	const char* m_Path;

};