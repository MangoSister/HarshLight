#version 450 core

in vec2 vs_Texcoord;
in vec3 vs_VoxelCoord;
in vec3 vs_WorldNormal;

out vec4 fragColor;

uniform sampler2D TexAlbedo;
layout (binding = 1, r32ui) coherent volatile uniform uimage3D TexVoxelAlbedo;
layout (binding = 2, r32ui) coherent volatile uniform uimage3D TexVoxelNormal;
//layout (binding = 3, r32ui) coherent volatile uniform uimage3D TexVoxelRadiance;
uniform sampler3D TexVoxelRadiance;
layout (binding = 4, rgba8) coherent volatile uniform image3D TexRadianceMipmap;
//uniform usampler3D TexVoxelAlbedo; //r32ui

layout (std140, binding = 0) uniform MainCamMtx
{
    mat4 View;
    mat4 Proj;
	vec4 CamWorldPos;
};

layout (std140, binding = 1) uniform VoxelCamMtx
{
    mat4 CamVoxelViewMtx;
    mat4 CamVoxelProjMtx;
	vec4 CamVoxelWorldPos;
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
	uint val = imageLoad(TexVoxelNormal, load_coord).x;
	vec4 dec_val = ColorUintToVec4(val);
	
	dec_val.xyz = dec_val.xyz * 2.0 - vec3(1.0);
	dec_val.xyz = normalize(dec_val.xyz);
	dec_val.xyz = 0.5 * (dec_val.xyz + vec3(1.0));

	fragColor = vec4(dec_val.xyz, 1.0);


	//vec4 radiance = imageLoad(TexRadianceMipmap, load_coord);
	//if(radiance.w > 0.0)
	//	radiance.xyz /= radiance.w;
	//else radiance.xyz = vec3(0.0);
	//radiance.w = 1.0;
	//fragColor = radiance;

	//fragColor = texelFetch(TexVoxelRadiance, load_coord, 0).rgba;
	//fragColor = texture(TexVoxelRadiance, vs_VoxelCoord);
	//fragColor.w = 1.0;
}