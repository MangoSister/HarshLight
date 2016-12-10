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
uniform sampler2D TexNormal;
uniform sampler2D TexSpecular;
uniform sampler2D TexOpacityMask;

uniform float Shininess;
uniform sampler2DShadow TexDirShadow[DIR_LIGHT_MAX_NUM];

vec3 ComputeDirLightBlinnPhong(in DirLight light, in vec3 view_dir, in vec3 normal, in float spec_scale)
{
	vec3 light_dir = normalize(-light.direction.xyz);
	vec3 half_dir = normalize(light_dir + view_dir);

	float diffuse_intensity = max(dot(normal, light_dir), 0.0);
	float spec_intensity = pow(max(dot(normal, half_dir), 0.0), Shininess) * spec_scale;

	return vec3(diffuse_intensity + spec_intensity) * light.color.xyz * light.color.w;
}

float ComputeDirLightShadow(uint idx)
{
	vec4 shadow_coord = DirLights[idx].lightProjMtx * DirLights[idx].lightMtx * vec4(vs_WorldPosition, 1.0);
	shadow_coord = shadow_coord * 0.5 + 0.5;
	shadow_coord.z *= SHADOW_BIAS;
	
	return textureProj(TexDirShadow[idx], shadow_coord);
}

vec3 ComputePointLightBlinnPhong(in PointLight light, in vec3 view_dir, in vec3 normal, in float spec_scale)
{
	float dist = length(light.position.xyz - vs_WorldPosition);
	float atten = light.color.w / (PointLightAtten.x + PointLightAtten.y * dist + PointLightAtten.z * dist * dist);
	if(atten < 0.001)
		return vec3(0.0);

	vec3 light_dir = (light.position.xyz - vs_WorldPosition) / dist;
	vec3 half_dir = normalize(light_dir + view_dir);

	float diffuse_intensity = max(dot(normal, light_dir), 0.0);
	float spec_intensity = pow(max(dot(normal, half_dir), 0.0), Shininess) * spec_scale;

	return vec3(diffuse_intensity + spec_intensity) * atten * light.color.xyz;
}

void main()
{
	float opacity_mask = texture(TexOpacityMask, vs_Texcoord).r;
	if(opacity_mask == 0.0)
		discard;
	
	fragColor = vec4(Ambient.xyz, 1.0);
	vec3 tan_normal = texture(TexNormal, vs_Texcoord).rgb;
	tan_normal.xy = tan_normal.xy * vec2(2.0) - vec2(1.0);
	tan_normal.z = sqrt(1.0 - dot(tan_normal.xy, tan_normal.xy));
	vec3 bitangent = cross(vs_WorldNormal, vs_WorldTangent);
	vec3 adj_world_normal = tan_normal.x * vs_WorldTangent + tan_normal.y * bitangent + tan_normal.z * vs_WorldNormal;
	vec3 view_dir = normalize(CamWorldPos.xyz - vs_WorldPosition);

	float spec_scale = texture(TexSpecular, vs_Texcoord).r;

    vec3 albedo = texture(TexAlbedo, vs_Texcoord).xyz;

	for(uint i = 0; i < ActiveDirLights; i++)
		fragColor.xyz += ComputeDirLightBlinnPhong(DirLights[i], view_dir, adj_world_normal, spec_scale) * ComputeDirLightShadow(i);
	
	for(uint i = 0; i < ActivePointLights; i++)
		fragColor.xyz += ComputePointLightBlinnPhong(PointLights[i], view_dir, adj_world_normal, spec_scale);
	
	fragColor.xyz *= albedo;
	//fragColor.xyz = vec3(spec_scale);
	//fragColor.w = 1.0;
	//fragColor = vec4(vs_WorldNormal * 0.5 + 0.5, 1.0);
} 