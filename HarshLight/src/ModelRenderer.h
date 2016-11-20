#pragma once

#include "Component.h"
#include "Model.h"
#include "Material.h"
#include <vector>

using namespace glm;

class ModelRenderer : public Component
{
public:
    explicit ModelRenderer(Model* model);

	void MoveTo(const glm::vec3& pos);
	void ScaleTo(const glm::vec3& scale);

    void Start() override;
    void Update(float dt) override;

    void AddMaterial(const Material* material);

private:

    Model* m_Model;
    std::vector<const Material*> m_Materials;
    mat4x4 m_Transform;
};