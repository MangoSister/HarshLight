#version 450 core

#define SHADOW_BIAS 0.98
#define SQRT3 1.73205080757
#define TAN30 0.57735026919
#define TAN10 0.1763269807
#define MAX_TRACE_DIST 200.0 //500, 1000?

in vec2 vs_Texcoord;
in vec3 vs_WorldPosition;
in vec3 vs_WorldNormal;
//in vec3 vs_VoxelCoord;
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
uniform sampler2D TexNormal;
uniform sampler2D TexSpecular;
uniform sampler2D TexOpacityMask;
uniform float Shininess;
uniform sampler2DShadow TexDirShadow[DIR_LIGHT_MAX_NUM];

uniform float VoxelDim;
uniform float VoxelScale;

uniform sampler3D ImgRadianceLeaf;
uniform sampler3D ImgRadianceInterior[6];

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

vec4 SampleVoxel(vec3 pos, float mip_level, vec3 dir)
{
	//leaf?
	//vec3 txc = (CamVoxelProjMtx * CamVoxelViewMtx * vec4(pos, 1.0)).xyz;
	//txc = txc * 0.5 + 0.5;
	
	vec4 sample_leaf = texture(ImgRadianceLeaf, pos);

	float interior_level = max(mip_level - 1.0, 0.0);
	vec4 sample_x = textureLod(ImgRadianceInterior[int((sign(dir.x) < 0))], pos, interior_level);
	vec4 sample_y = textureLod(ImgRadianceInterior[2 + int((sign(dir.y) < 0))], pos, interior_level);
	vec4 sample_z = textureLod(ImgRadianceInterior[4 + int((sign(dir.z) < 0))], pos, interior_level);

	vec3 weights = abs(dir);
	float inv_sum = 1.0 / (weights.x + weights.y + weights.z);
	weights *= inv_sum;

	vec4 sample_interior = sample_x * weights.x + sample_y * weights.y + sample_z * weights.z;
	
	vec4 res = mix(sample_leaf, sample_interior, clamp(mip_level, 0, 1));
	return res;
}

vec3 VoxelConeTracing(vec3 origin, vec3 dir, float half_tan, float max_dist)
{
	vec4 accum = vec4(0.0, 0.0, 0.0, 0.0);
	float diameter = VoxelScale;
	float dist = 1.0 * VoxelScale / half_tan;
	float diameter_init = diameter;
	float mip_level = 0;
	vec4 sample_val = vec4(0.0, 0.0, 0.0, 0.0);

	vec3 sample_pos = origin + dist * dir;
	vec3 voxel_coord = ((CamVoxelProjMtx * CamVoxelViewMtx * vec4(sample_pos, 1.0)).xyz) * vec3(0.5) + vec3(0.5);
	int step_count = 0;
	while(dist < max_dist && accum.w < 1.0 && 
		all(greaterThanEqual(voxel_coord, vec3(0.0))) && 
		all(lessThanEqual(voxel_coord, vec3(1.0))))  //make sure sample_pos in bound
	{
		diameter = max(diameter, 2.0 * half_tan * dist);
		mip_level = log2(diameter / VoxelScale);
		sample_val = SampleVoxel(voxel_coord, mip_level, dir);

		if(sample_val.w > 0.0)
		{
			sample_val.xyz /= sample_val.w;
			sample_val.w = 1.0 - pow(1.0 - sample_val.w, diameter / diameter_init);
			//alpha blend (pre-multiply)
			sample_val.xyz *= sample_val.w;
			accum += (1 - accum.w) * sample_val;
		}

		dist += diameter * 0.5;
		sample_pos = origin + dist * dir;
		voxel_coord = ((CamVoxelProjMtx * CamVoxelViewMtx * vec4(sample_pos, 1.0)).xyz) * vec3(0.5) + 0.5;
	}

	return accum.xyz; //divide by w causes error
} 

void main()
{
	float opacity_mask = texture(TexOpacityMask, vs_Texcoord).r;
	if(opacity_mask == 0.0)
		discard;

	//fragColor = vec4(Ambient.xyz, 1.0);
	fragColor = vec4(0, 0, 0, 1);
	vec3 albedo = texture(TexAlbedo, vs_Texcoord).xyz;
	vec3 tan_normal = texture(TexNormal, vs_Texcoord).rgb;
	tan_normal.xy = tan_normal.xy * vec2(2.0) - vec2(1.0);
	tan_normal.z = sqrt(1.0 - dot(tan_normal.xy, tan_normal.xy));
	vec3 bitangent = cross(vs_WorldNormal, vs_WorldTangent);
	vec3 adj_world_normal = tan_normal.x * vs_WorldTangent + tan_normal.y * bitangent + tan_normal.z * vs_WorldNormal;
	vec3 adj_world_tangent = normalize(cross(adj_world_normal, vs_WorldTangent));
	vec3 adj_world_bitangent = cross(adj_world_normal, vs_WorldTangent);

	vec3 view_dir = normalize(CamWorldPos.xyz - vs_WorldPosition);
	float spec_scale = texture(TexSpecular, vs_Texcoord).r;

	/* ----------------- Direct Lighting --------------------- */
	for(uint i = 0; i < ActiveDirLights; i++)
		fragColor.xyz += ComputeDirLightBlinnPhong(DirLights[i], view_dir, adj_world_normal, spec_scale) * ComputeDirLightShadow(i);
	
	for(uint i = 0; i < ActivePointLights; i++)
		fragColor.xyz += ComputePointLightBlinnPhong(PointLights[i], view_dir, adj_world_normal, spec_scale);	
	/* ------------------------------------------------------ */

	////indirect diffuse
	fragColor.xyz += 0.2 * VoxelConeTracing(vs_WorldPosition, adj_world_normal, TAN30, MAX_TRACE_DIST);
	//cos(60) = 0.5
	fragColor.xyz += 0.1 * VoxelConeTracing(vs_WorldPosition, normalize(adj_world_normal + SQRT3 * adj_world_tangent), TAN30, MAX_TRACE_DIST);
	fragColor.xyz += 0.1 * VoxelConeTracing(vs_WorldPosition, normalize(adj_world_normal - SQRT3 * adj_world_tangent), TAN30, MAX_TRACE_DIST);
	fragColor.xyz += 0.1 * VoxelConeTracing(vs_WorldPosition, normalize(adj_world_normal + SQRT3 * adj_world_bitangent), TAN30, MAX_TRACE_DIST);
	fragColor.xyz += 0.1 * VoxelConeTracing(vs_WorldPosition, normalize(adj_world_normal - SQRT3 * adj_world_bitangent), TAN30, MAX_TRACE_DIST);

	//indirect specular
	vec3 ref_dir = -reflect(view_dir, adj_world_normal);
	float spec_half_tan = tan(0.5 * acos(pow(0.244, 1.0 / (1 + Shininess)))); //magic number here, from GPU PRO5 SSR cone trace chapter
	//precompute this

	fragColor.xyz += spec_scale * VoxelConeTracing(vs_WorldPosition, ref_dir, spec_half_tan, MAX_TRACE_DIST);
	//voxel shadow ??
	
	fragColor.xyz *= albedo;
	fragColor.xyz = sqrt(fragColor.xyz); // approximate gamma correction (2 instead of 2.2)
}