#version 450 core

in vec2 vs_Texcoord;
in vec3 vs_VoxelCoord;
in vec3 vs_WorldNormal;

out vec4 fragColor;

uniform sampler2D TexAlbedo;
layout (binding = 1, r32ui) coherent volatile uniform uimage3D TexVoxel;
layout (binding = 2, r32ui) coherent volatile uniform uimage3D TexVoxelNormal;
//uniform usampler3D TexVoxel; //r32ui

layout (std140, binding = 0) uniform MainCamMtx
{
    mat4 View;
    mat4 Proj;
};

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
	ivec3 dim = imageSize(TexVoxelNormal);
	vec3 dim_v = vec3(float(dim.x), float(dim.y), float(dim.z));
	ivec3 load_coord = ivec3(dim_v * vs_VoxelCoord);
	uvec4 val = imageLoad(TexVoxelNormal, load_coord);
	vec4 dec_val = ColorUintToVec4(val.x);
	dec_val.xyz = dec_val.xyz * 2.0 - vec3(1.0);
	dec_val.xyz = normalize(dec_val.xyz);
	dec_val.xyz = 0.5 * (dec_val.xyz + vec3(1.0));
	//dec_val.xyz = ((View * vec4(dec_val.xyz, 0.0)).xyz);
	//dec_val.xyz = 0.5 * (dec_val.xyz + vec3(1.0));
	fragColor = vec4(dec_val.xyz, 1.0);
	//fragColor = vec4(dec_val.w * 20);
	//fragColor.w = 1.0;
	//fragColor = dec_val;
}