#version 450 core
layout (location = 0) out vec4 GPositionAndSpecPower;
layout (location = 1) out vec4 GNormalAndTangent;
layout (location = 2) out vec4 GAlbedoAndSpecIntensity;

in vec2 vs_Texcoord;
in vec3 vs_WorldPosition;
in vec3 vs_WorldNormal;
in vec3 vs_WorldTangent;

uniform float Shininess;

uniform sampler2D TexAlbedo;
uniform sampler2D TexNormal;
uniform sampler2D TexSpecular;
uniform sampler2D TexOpacityMask;

void main()
{
	float opacity_mask = texture(TexOpacityMask, vs_Texcoord).r;
	if(opacity_mask == 0.0)
		discard; //alpha cutout

	GPositionAndSpecPower.xyz = vs_WorldPosition;
	GPositionAndSpecPower.w = Shininess;

	//normal and tangent are adjusted by normal map
	vec3 tan_normal = texture(TexNormal, vs_Texcoord).rgb;
	tan_normal.xy = tan_normal.xy * vec2(2.0) - vec2(1.0);
	tan_normal.z = sqrt(1.0 - dot(tan_normal.xy, tan_normal.xy));
	vec3 bitangent = cross(vs_WorldNormal, vs_WorldTangent);
	vec3 adj_world_normal = tan_normal.x * vs_WorldTangent + tan_normal.y * bitangent + tan_normal.z * vs_WorldNormal;
	vec3 adj_world_tangent = normalize(cross(adj_world_normal, vs_WorldTangent));

	GNormalAndTangent.xy = adj_world_normal.xy;
	GNormalAndTangent.zw = adj_world_tangent.xy;

	GAlbedoAndSpecIntensity.xyz = texture(TexAlbedo, vs_Texcoord).xyz;
	GAlbedoAndSpecIntensity.w = texture(TexSpecular, vs_Texcoord).r;
}