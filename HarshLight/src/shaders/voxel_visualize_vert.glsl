#version 450 core

layout (location = 0) in vec3 app_Position;
layout (location = 1) in vec3 app_Normal;
layout (location = 2) in vec2 app_Uv;

out vec2 vs_Texcoord;
out vec3 vs_VoxelCoord;
out vec3 vs_WorldNormal;

layout (std140, binding = 0) uniform MainCamMtx
{
    mat4 View;
    mat4 Proj;
	vec3 CamWorldPos;
};

layout (std140, binding = 1) uniform VoxelCamMtx
{
    mat4 CamVoxelViewMtx;
    mat4 CamVoxelProjMtx;
	vec3 CamVoxelWorldPos;
};

uniform mat4 Model;

void main()
{
    gl_Position =  Proj * View * Model * vec4(app_Position, 1.0);
	vs_Texcoord = app_Uv;
	vs_VoxelCoord = (CamVoxelProjMtx * CamVoxelViewMtx * Model * vec4(app_Position, 1.0)).xyz;
	//voxel_space_pos.xyz is in NDC space now
	vs_VoxelCoord = (vs_VoxelCoord + vec3(1.0, 1.0, 1.0)) * 0.5;
	//voxel_space_pos.xyz is in [0,1]^3 space now
	vs_WorldNormal = (transpose(inverse(Model)) * vec4(app_Normal, 0.0)).xyz;
	vs_WorldNormal = normalize(vs_WorldNormal);
}