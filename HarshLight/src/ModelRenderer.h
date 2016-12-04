#pragma once

#include "Model.h"
#include "Material.h"
#include <vector>

using namespace glm;

typedef uint8_t RenderPassFlag;
namespace RenderPass
{
    enum
    {
        kNone = 0x00,
        kVoxelize = 0x01,
		kLightInjection = 0x02,
        kRegular = 0x04, 
		kPost = 0x08,
        kCount = 4,
    };
}

class ModelRenderer
{
public:
    explicit ModelRenderer(Model* model);
	virtual ~ModelRenderer() { }
	void MoveTo(const glm::vec3& pos);
	void ScaleTo(const glm::vec3& scale);

	void Render(RenderPassFlag pass);

    void AddMaterial(RenderPassFlag pass, Material* material);
    const std::vector<Material*>& GetMaterial(RenderPassFlag pass) const;
	void SetRenderPass(RenderPassFlag flag);
	RenderPassFlag GetRenderPass() const;

    const mat4x4& GetTransform() const;

protected:

	mat4x4 m_Transform;
    Model* m_Model;

	RenderPassFlag m_RenderPassFlag;
	std::vector<Material*> m_VoxelizeMaterials;
	std::vector<Material*> m_LightInjectionMaterials;
    std::vector<Material*> m_Materials;	
	std::vector<Material*> m_PostMaterials;

};