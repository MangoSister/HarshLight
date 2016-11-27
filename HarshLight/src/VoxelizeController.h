#pragma once

#include "Material.h"
#include "Component.h"

class VoxelizeController : public Component
{
public: 
    explicit VoxelizeController(uint32_t dim);
    void Start() override;
    void Update(float dt) override { }
    
    void SetVoxelDim(uint32_t dim);
    uint32_t GetVoxelDim() const;

private:
    
    static const char* s_VoxelDimName;

    uint32_t m_VoxelDim;
};