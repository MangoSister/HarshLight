#version 450 core

in vec2 vs_Texcoord;
in vec3 vs_VoxelCoord;
in vec3 vs_WorldNormal;

out vec4 fragColor;

uniform sampler2D TexAlbedo;
layout (binding = 1, r32ui) coherent volatile uniform uimage3D TexVoxel;
layout (binding = 2, r32ui) coherent volatile uniform uimage3D TexVoxelNormal;
//uniform usampler3D TexVoxel; //r32ui

layout (std140, binding = 1) uniform VoxelCamMtx
{
    mat4 CamVoxelViewMtx;
    mat4 CamVoxelProjMtx;
};


vec4 ColorUintToVec4(uint val) 
{
	float r = float((val & 0xFF000000) >> 24U);
	float g = float((val & 0x00FF0000) >> 16U);
	float b = float((val & 0x0000FF00) >> 8U);
	float a = float((val & 0x000000FF));

	vec4 o = vec4(r, g, b, a);
	o /= 255.0;
	o = clamp(o, vec4(0.0, 0.0, 0.0, 0.0), vec4(1.0, 1.0, 1.0, 1.0));
	return o;
}

void main()
{
	ivec3 dim = imageSize(TexVoxel);
	vec3 dim_v = vec3(float(dim.x), float(dim.y), float(dim.z));
	ivec3 load_coord = ivec3(dim_v * vs_VoxelCoord);
	uvec4 val = imageLoad(TexVoxel, load_coord);
	fragColor = vec4(ColorUintToVec4(val.x).xyz, 1.0);
}