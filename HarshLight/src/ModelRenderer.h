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
        kRegular = 0x02,
        kPost = 0x04,
        kCount = 3,
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

    void AddMaterial(RenderPassFlag pass, const Material* material);

	void SetRenderPass(RenderPassFlag flag);
	RenderPassFlag GetRenderPass() const;

protected:

	mat4x4 m_Transform;
    Model* m_Model;

	RenderPassFlag m_RenderPassFlag;
    std::vector<const Material*> m_Materials;
	std::vector<const Material*> m_VoxelizeMaterials;
	std::vector<const Material*> m_PostMaterials;

};