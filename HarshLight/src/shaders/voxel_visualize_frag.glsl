#version 450 core

in vec2 vs_Texcoord;
in vec3 vs_VoxelCoord;
in vec3 vs_WorldNormal;

out vec4 fragColor;

uniform sampler2D TexAlbedo;
uniform sampler3D TexVoxel;

layout (std140, binding = 1) uniform VoxelCamMtx
{
    mat4 CamVoxelViewMtx;
    mat4 CamVoxelProjMtx;
};

void main()
{
	fragColor = texture(TexVoxel, vs_VoxelCoord);
}