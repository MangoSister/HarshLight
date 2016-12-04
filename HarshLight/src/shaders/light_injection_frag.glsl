#version 450 core

in vec2 vs_Texcoord;
in vec3 vs_VoxelCoord;
in vec3 vs_WorldNormal;

out vec4 fragColor;

layout (binding = 1, r32ui) coherent volatile uniform uimage3D TexVoxelAlbedo;
layout (binding = 2, r32ui) coherent volatile uniform uimage3D TexVoxelNormal;
layout (binding = 3, r32ui) coherent volatile uniform uimage3D TexVoxelRadiance;

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

/* --------------  Lighting Info  ----------- */
#define DIR_LIGHT_MAX_NUM 4
#define POINT_LIGHT_MAX_NUM 4

struct DirLight
{
	vec4 direction;
	vec4 color;
	mat4 lightMtx;
};

struct PointLight
{
	vec4 position;
	vec4 color;
};

layout (std140, binding = 2) uniform LightInfo
{
	DirLight DirLights[DIR_LIGHT_MAX_NUM];
	PointLight PointLights[POINT_LIGHT_MAX_NUM];
	vec4 Ambient;
	vec4 PointLightAtten;
	uint ActiveDirLights;
	uint ActivePointLights;
};
/* ---------------------------------------- */

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

uint ColorVec4ToUint(vec4 val) 
{
	val *= 255.0;
	return	(uint(val.x) & 0x000000FF) << 24U | 
			(uint(val.y) & 0x000000FF) << 16U | 
			(uint(val.z) & 0x000000FF) << 8U | 
			(uint(val.w) & 0x000000FF);
}

vec3 ComputeDirLightLambertian(DirLight light, vec3 world_pos, vec3 albedo, vec3 normal)
{
	vec3 light_dir = normalize(-light.direction.xyz);
	float diffuse_intensity = max(dot(normalize(normal), light_dir), 0.0);

	return vec3(diffuse_intensity) * light.color.xyz * light.color.w;
}

vec3 ComputePointLightLambertian(PointLight light, vec3 world_pos, vec3 albedo, vec3 normal)
{
	float dist = length(light.position.xyz - world_pos);
	float atten = light.color.w / (PointLightAtten.x + PointLightAtten.y * dist + PointLightAtten.z * dist * dist);
	if(atten < 0.001)
		return vec3(0.0);
	
	vec3 light_dir = (light.position.xyz - world_pos) / dist;
	float diffuse_intensity = max(dot(normalize(normal), light_dir), 0.0);

	return vec3(diffuse_intensity) * atten * light.color.xyz;
}

void main()
{
	ivec3 dim = imageSize(TexVoxelAlbedo);
	vec3 dim_v = vec3(float(dim.x), float(dim.y), float(dim.z));
	ivec3 load_coord = ivec3(dim_v * vs_VoxelCoord);

	uint albedo_enc = imageLoad(TexVoxelAlbedo, load_coord).x;
	vec4 albedo_dec = ColorUintToVec4(albedo_enc);
	uint normal_enc = imageLoad(TexVoxelNormal, load_coord).x;
	vec4 normal_dec = ColorUintToVec4(normal_enc);
	normal_dec.xyz = normal_dec.xyz * 2.0 - vec3(1.0);

	//get voxel center
	vec3 voxel_center = vec3(load_coord) + vec3(0.5);
	voxel_center /= dim_v; //[0, 1]^3
	voxel_center = voxel_center * vec3(2.0) - vec3(1.0); //[-1, 1]^3, NDC
	voxel_center = (inverse(CamVoxelProjMtx * CamVoxelViewMtx) * vec4(voxel_center, 1.0)).xyz; // world

	fragColor = vec4(0.0, 0.0, 0.0, 1.0); // no ambient
	//radiance (only diffuse)
	for(uint i = 0; i < ActiveDirLights; i++)
		fragColor.xyz += ComputeDirLightLambertian(DirLights[i], voxel_center, albedo_dec.xyz, normal_dec.xyz);
	
	for(uint i = 0; i < ActivePointLights; i++)
		fragColor.xyz += ComputePointLightLambertian(PointLights[i], voxel_center, albedo_dec.xyz, normal_dec.xyz);

	fragColor.xyz *= albedo_dec.xyz;

	//simply sample voxel center point once
	if((imageLoad(TexVoxelRadiance, load_coord).x & 1) == 0)
	{
		uint u32_fragColor = ColorVec4ToUint(fragColor);
		imageAtomicCompSwap(TexVoxelRadiance, load_coord, 0, u32_fragColor);
	}
}