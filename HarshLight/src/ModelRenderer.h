#pragma once

#include "Component.h"
#include "Model.h"
#include "Material.h"


using namespace glm;

class ModelRenderer : public Component
{
public:
    explicit ModelRenderer(Model* model, Material* material);

	void MoveTo(const glm::vec3& pos);
	void ScaleTo(const glm::vec3& scale);

    void Start() override;
    void Update(float dt) override;
private:

    Model* m_Model;
    Material* m_Material;
    mat4x4 m_Transform;
};