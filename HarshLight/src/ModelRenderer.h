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
        kDirLightInjection = 0x02,
        kPointLightInjection = 0x04,
        kGeometry = 0x08,
        kDeferredIndirectDiffuse = 0x10,
		kDeferredFinalComposition = 0x20,
		kPost = 0x40,
        kCount = 7,
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
    void Render(RenderPassFlag pass, const glm::vec3& center, float radius);
    void AddMaterial(RenderPassFlag pass, Material* material);
    const std::vector<Material*>& GetMaterial(RenderPassFlag pass) const;
	void SetRenderPass(RenderPassFlag flag);
	RenderPassFlag GetRenderPass() const;

    const mat4x4& GetTransform() const;

    inline const Model* GetModel() const
    { return m_Model; }

protected:

	mat4x4 m_Transform;
    Model* m_Model;

	RenderPassFlag m_RenderPassFlag;
	std::vector<Material*> m_VoxelizeMaterials;
	std::vector<Material*> m_DirLightInjectionMaterials;
    std::vector<Material*> m_PointLightInjectionMaterials;
    std::vector<Material*> m_GeometryPassMaterial;	
	std::vector<Material*> m_DeferredIndirectDiffuseMaterial;
    std::vector<Material*> m_DeferredFinalComposition;
	std::vector<Material*> m_PostMaterials;
};