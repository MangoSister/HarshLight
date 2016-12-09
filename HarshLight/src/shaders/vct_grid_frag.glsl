#version 450 core

#define SHADOW_BIAS 0.98
#define SQRT3 1.73205080757
#define TAN30 0.57735026919
#define MAX_TRACE_DIST 1000.0

in vec2 vs_Texcoord;
in vec3 vs_WorldPosition;
in vec3 vs_WorldNormal;
in vec3 vs_VoxelCoord;
in vec3 vs_WorldTangent;

out vec4 fragColor;

/* --------------  Lighting Info  ----------- */
#define DIR_LIGHT_MAX_NUM 4
#define POINT_LIGHT_MAX_NUM 4

struct DirLight
{
	vec4 direction;
	vec4 color;
	mat4 lightMtx;
	mat4 lightProjMtx;
};

struct PointLight
{
	vec4 position;
	vec4 color;
};

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

layout (std140, binding = 2) uniform LightInfo
{
	DirLight DirLights[DIR_LIGHT_MAX_NUM];
	PointLight PointLights[POINT_LIGHT_MAX_NUM];
	vec4 Ambient;
	vec4 PointLightAtten;
	uint ActiveDirLights;
	uint ActivePointLights;
};
/* --------------------------------------- */

uniform sampler2D TexAlbedo;
uniform float Shininess;
uniform sampler2DShadow TexDirShadow[DIR_LIGHT_MAX_NUM];

uniform float VoxelDim;
uniform float VoxelScale;

uniform sampler3D ImgRadianceLeaf;
uniform sampler3D ImgRadianceInterior[6];

vec3 ComputeDirLightBlinnPhong(DirLight light)
{
	vec3 light_dir = normalize(-light.direction.xyz);
	vec3 view_dir = normalize(CamWorldPos.xyz - vs_WorldPosition);
	vec3 half_dir = normalize(light_dir + view_dir);

	float diffuse_intensity = max(dot(vs_WorldNormal, light_dir), 0.0);
	float spec_intensity = 0;//pow(max(dot(vs_WorldNormal, half_dir), 0.0), Shininess);

	return vec3(diffuse_intensity + spec_intensity) * light.color.xyz * light.color.w;
}

float ComputeDirLightShadow(uint idx)
{
	vec4 shadow_coord = DirLights[idx].lightProjMtx * DirLights[idx].lightMtx * vec4(vs_WorldPosition, 1.0);
	shadow_coord = shadow_coord * 0.5 + 0.5;
	shadow_coord.z *= SHADOW_BIAS;
	
	return textureProj(TexDirShadow[idx], shadow_coord);
}

vec3 ComputePointLightBlinnPhong(PointLight light)
{
	float dist = length(light.position.xyz - vs_WorldPosition);
	float atten = light.color.w / (PointLightAtten.x + PointLightAtten.y * dist + PointLightAtten.z * dist * dist);
	if(atten < 0.001)
		return vec3(0.0);

	vec3 light_dir = (light.position.xyz - vs_WorldPosition) / dist;
	vec3 view_dir = normalize(CamWorldPos.xyz - vs_WorldPosition);
	vec3 half_dir = normalize(light_dir + view_dir);

	float diffuse_intensity = max(dot(vs_WorldNormal, light_dir), 0.0);
	float spec_intensity = 0;//pow(max(dot(vs_WorldNormal, half_dir), 0.0), Shininess);

	return vec3(diffuse_intensity + spec_intensity) * atten * light.color.xyz;
}

vec4 SampleVoxel(vec3 pos, float mip_level, vec3 dir)
{
	//leaf?
	vec3 txc = (CamVoxelProjMtx * CamVoxelViewMtx * vec4(pos, 1.0)).xyz;
	txc = txc * 0.5 + 0.5;
	
	vec4 sample_leaf = texture(ImgRadianceLeaf, txc);

	float interior_level = max(mip_level - 1.0, 0.0);
	vec4 sample_x = textureLod(ImgRadianceInterior[int((sign(dir.x) < 0))], txc, interior_level);
	vec4 sample_y = textureLod(ImgRadianceInterior[2 + int((sign(dir.y) < 0))], txc, interior_level);
	vec4 sample_z = textureLod(ImgRadianceInterior[4 + int((sign(dir.z) < 0))], txc, interior_level);

	vec3 weights = abs(dir);
	float inv_sum = 1.0 / (weights.x + weights.y + weights.z);
	weights *= inv_sum;

	vec4 sample_interior = sample_x * weights.x + sample_y * weights.y + sample_z * weights.z;
	
	vec4 res = sample_interior;//mix(sample_leaf, sample_interior, clamp(mip_level, 0, 1));
	return res;
}

vec3 VoxelConeTracing(vec3 origin, vec3 dir, float half_tan, float max_dist)
{
	vec3 sample_pos = origin;
	
	vec4 accum = vec4(0.0, 0.0, 0.0, 0.0);
	float diameter = VoxelScale;
	float dist = 2.0 * VoxelScale / half_tan;
	float diameter_init = diameter;
	float mip_level = 0;
	vec4 sample_val = vec4(0.0, 0.0, 0.0, 0.0);

	while(dist < max_dist && accum.w < 1.0)
	{
		sample_pos = origin + dist * dir;
		diameter = max(diameter, 2.0 * half_tan * dist);
		mip_level = log2(diameter / VoxelScale);
		sample_val = SampleVoxel(sample_pos, mip_level, dir);

		sample_val.xyz /= sample_val.w;
		sample_val.w = 1.0 - pow(1.0 - sample_val.w, diameter / diameter_init);
		//alpha blend (pre-multiply)
		sample_val.xyz *= sample_val.w;
		accum += (1 - accum.w) * sample_val;

		dist += diameter;

	}

	return accum.xyz / accum.w;
} 

void main()
{
	//fragColor = vec4(Ambient.xyz, 1.0);
	fragColor = vec4(0, 0, 0, 1);

	/* ----------------- Direct Lighting --------------------- */
    vec3 albedo = texture(TexAlbedo, vs_Texcoord).xyz;

	for(uint i = 0; i < ActiveDirLights; i++)
		fragColor.xyz += ComputeDirLightBlinnPhong(DirLights[i]) * ComputeDirLightShadow(i);
	
	for(uint i = 0; i < ActivePointLights; i++)
		fragColor.xyz += ComputePointLightBlinnPhong(PointLights[i]);
	
	fragColor.xyz *= albedo;
	/* ------------------------------------------------------ */

	//indirect diffuse

	fragColor.xyz += 0.2 * VoxelConeTracing(vs_WorldPosition, vs_WorldNormal, TAN30, MAX_TRACE_DIST);
	vec3 bitangent = cross(vs_WorldNormal, vs_WorldTangent);
	//////cos(60) = 0.5
	fragColor.xyz += 0.1 * VoxelConeTracing(vs_WorldPosition, normalize(vs_WorldNormal + SQRT3 * vs_WorldTangent), TAN30, MAX_TRACE_DIST);
	fragColor.xyz += 0.1 * VoxelConeTracing(vs_WorldPosition, normalize(vs_WorldNormal - SQRT3 * vs_WorldTangent), TAN30, MAX_TRACE_DIST);
	fragColor.xyz += 0.1 * VoxelConeTracing(vs_WorldPosition, normalize(vs_WorldNormal + SQRT3 * bitangent), TAN30, MAX_TRACE_DIST);
	fragColor.xyz += 0.1 * VoxelConeTracing(vs_WorldPosition, normalize(vs_WorldNormal - SQRT3 * bitangent), TAN30, MAX_TRACE_DIST);

	//indirect specular

	//voxel shadow ??

}