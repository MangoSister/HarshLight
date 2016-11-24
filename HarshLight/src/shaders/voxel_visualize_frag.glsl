#version 450 core

in vec2 vs_Texcoord;
in vec3 vs_WorldPosition;
in vec3 vs_WorldNormal;

out vec4 fragColor;

uniform sampler2D TexAlbedo;
uniform sampler3D TexVoxel;

layout (std140, binding = 1) uniform VoxelCamMtx
{
    mat4 CamVoxelViewMtx;
    mat4 CamVoxelProjMtx;
};

//uniform mat4 CamVoxelViewMtx;
//uniform mat4 CamVoxelProjMtx;
uniform float VoxelDim;

void main()
{
	vec4 voxel_space_pos = CamVoxelProjMtx * CamVoxelViewMtx * vec4(vs_WorldPosition, 1.0);
	voxel_space_pos /= voxel_space_pos.w;
	//voxel_space_pos.xyz is in NDC space now
	voxel_space_pos.xyz = (voxel_space_pos.xyz + 1) * 0.5; 
	//voxel_space_pos.xyz is in [0,1]^3 space now

	fragColor = texture(TexVoxel, voxel_space_pos.xyz);
}