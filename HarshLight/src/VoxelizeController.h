#pragma once

#include "Camera.h"
#include "Material.h"
#include "Component.h"

class VoxelizeController : public Component
{
public: 
    explicit VoxelizeController(uint32_t dim, float extent, Camera* voxel_cam);
    void Start() override;
    void Update(float dt) override { }
    
    void SetVoxelDim(uint32_t dim);
    uint32_t GetVoxelDim() const;

private:
    
    static const char* s_VoxelDimName;
    static const char* s_ViewMtxToDownName;
    static const char* s_ViewMtxToLeftName;
    static const char* s_ViewMtxToForwardName;

    uint32_t m_VoxelDim;
    float m_Extent;
    Camera* m_VoxelCam;
};