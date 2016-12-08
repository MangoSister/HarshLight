#version 450 core

#define SHADOW_BIAS 0.98

in vec2 vs_Texcoord;
in vec3 vs_WorldPosition;
in vec3 vs_WorldNormal;
in vec3 vs_WorldTangent;

out vec4 fragColor;

layout (std140, binding = 0) uniform MainCamMtx
{
    mat4 View;
    mat4 Proj;
	vec4 CamWorldPos;
};

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

vec3 ComputeDirLightBlinnPhong(DirLight light)
{
	vec3 light_dir = normalize(-light.direction.xyz);
	vec3 view_dir = normalize(CamWorldPos.xyz - vs_WorldPosition);
	vec3 half_dir = normalize(light_dir + view_dir);

	float diffuse_intensity = max(dot(vs_WorldNormal, light_dir), 0.0);
	float spec_intensity = pow(max(dot(vs_WorldNormal, half_dir), 0.0), Shininess);

	return vec3(diffuse_intensity + spec_intensity) * light.color.xyz * light.color.w;
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
	float spec_intensity = pow(max(dot(vs_WorldNormal, half_dir), 0.0), Shininess);

	return vec3(diffuse_intensity + spec_intensity) * atten * light.color.xyz;
}

float ComputeDirLightShadow(uint idx)
{
	vec4 shadow_coord = DirLights[idx].lightProjMtx * DirLights[idx].lightMtx * vec4(vs_WorldPosition, 1.0);
	shadow_coord = shadow_coord * 0.5 + 0.5;
	shadow_coord.z *= SHADOW_BIAS;
	
	return textureProj(TexDirShadow[idx], shadow_coord);
}

void main()
{
	fragColor = vec4(Ambient.xyz, 1.0);

    vec3 albedo = texture(TexAlbedo, vs_Texcoord).xyz;

	for(uint i = 0; i < ActiveDirLights; i++)
		fragColor.xyz += ComputeDirLightBlinnPhong(DirLights[i]) * ComputeDirLightShadow(i);
	
	for(uint i = 0; i < ActivePointLights; i++)
		fragColor.xyz += ComputePointLightBlinnPhong(PointLights[i]);
	
	fragColor.xyz *= albedo;

	//fragColor = vec4(vs_WorldNormal * 0.5 + 0.5, 1.0);
} 