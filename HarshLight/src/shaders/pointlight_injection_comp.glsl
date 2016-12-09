#version 450 core

#define LOCAL_SIZE 8
layout(local_size_x = LOCAL_SIZE, local_size_y = LOCAL_SIZE) in;

layout (binding = 0) uniform samplerCube cubeDepthMap;
layout (binding = 1, r32ui) coherent volatile uniform uimage3D TexVoxelAlbedo;
layout (binding = 2, r32ui) coherent volatile uniform uimage3D TexVoxelNormal;
layout (binding = 3, r32ui) coherent volatile uniform uimage3D TexVoxelRadiance;

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
	mat4 lightProjMtx;
};

struct PointLight
{
	vec4 position;
	vec4 color;
};

//must first update light info before run light injection
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

layout (std140, binding = 3) uniform LightCapture
{
	mat4 CubeLightMtx[6];
	mat4 CubeLightProjMtx;
	vec4 PointLightWorldPos;
	vec2 CaptureRange;
};

uniform PointLight CurrPointLight;

uniform uint CurrLightPass;//start from 1, max 255

vec3 ComputePointLightLambertian(PointLight light, vec3 world_pos, vec3 albedo, vec3 normal)
{
	float dist = length(light.position.xyz - world_pos);
	float atten = light.color.w / (PointLightAtten.x + PointLightAtten.y * dist + PointLightAtten.z * dist * dist);
	
	vec3 light_dir = (light.position.xyz - world_pos) / dist;
	float diffuse_intensity = max(dot(normalize(normal), light_dir), 0.0);

	return vec3(diffuse_intensity) * atten * light.color.xyz;
}

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

void main()
{
	const uvec2 map_dim = gl_NumWorkGroups.xy * uvec2(gl_WorkGroupSize.x, gl_WorkGroupSize.y);
	vec2 map_coord;
	map_coord.x = float(gl_GlobalInvocationID.x) / float(map_dim.x);
	map_coord.y = float(gl_GlobalInvocationID.y) / float(map_dim.y);
	vec3 faceUVs[6];
	faceUVs[0] = vec3( 1.0, -map_coord.y, -map_coord.x);
	faceUVs[1] = vec3(-1.0, -map_coord.y,  map_coord.x);
	faceUVs[2] = vec3( map_coord.x,  1.0, map_coord.y);
	faceUVs[3] = vec3(-map_coord.x, -1.0, map_coord.y);
	faceUVs[4] = vec3( map_coord.x, -map_coord.y,  1.0);
	faceUVs[5] = vec3(-map_coord.x, -map_coord.y, -1.0);

	for(uint face = 0; face < 6; face++)
	{
		vec3 voxel_coord;
		voxel_coord.xy = map_coord;
		voxel_coord.z = texture(cubeDepthMap, faceUVs[face]).r; //linear depth
		if(voxel_coord.z >= 1.0)
			continue;
		//restore to proj space z
		voxel_coord.z = mix(CaptureRange.x, CaptureRange.y, voxel_coord.z);

		voxel_coord.xy = 2.0 * voxel_coord.xy - vec2(1.0);
		//xy now [-1, 1]^3

		voxel_coord.xy *= voxel_coord.z;
		voxel_coord.z *= -1;
		float w = -voxel_coord.z;
		voxel_coord.z = (CaptureRange.x + CaptureRange.y + 2 * CaptureRange.x * CaptureRange.y / voxel_coord.z) / (CaptureRange.y - CaptureRange.x);
		voxel_coord.z *= w;

		voxel_coord = (inverse(CubeLightProjMtx * CubeLightMtx[face]) * vec4(voxel_coord, w)).xyz;
		//now world space

		voxel_coord = (CamVoxelProjMtx * CamVoxelViewMtx * vec4(voxel_coord, 1.0)).xyz;
		//now in voxel space [-1, 1]^3

		voxel_coord = 0.5 * (voxel_coord + vec3(1.0));
		//now in voxel space [0, 1]^3

		if(all(greaterThanEqual(voxel_coord, vec3(0.0))) &&
		   all(lessThan(voxel_coord, vec3(1.0))) )
	   {
			vec3 voxel_dim = vec3(imageSize(TexVoxelAlbedo).xxx);
			ivec3 load_coord = ivec3(voxel_coord * voxel_dim);
		
			uint albedo_enc = imageLoad(TexVoxelAlbedo, load_coord).x;
			//skip potential empty voxel though there shouldn't be ideally?
			//if(albedo_enc == 0)
			//	continue;
			vec4 albedo_dec = ColorUintToVec4(albedo_enc);
			uint normal_enc = imageLoad(TexVoxelNormal, load_coord).x;
			vec4 normal_dec = ColorUintToVec4(normal_enc);
			normal_dec.xyz = normal_dec.xyz * 2.0 - vec3(1.0);
			normal_dec.xyz = normalize(normal_dec.xyz);
			//get voxel center
			vec3 voxel_center = vec3(load_coord) + vec3(0.5);
			voxel_center /= voxel_dim; //[0, 1]^3
			voxel_center = voxel_center * vec3(2.0) - vec3(1.0); //[-1, 1]^3, NDC
			voxel_center = (inverse(CamVoxelProjMtx * CamVoxelViewMtx) * vec4(voxel_center, 1.0)).xyz; // world
			//radiance (only diffuse)
			vec4 radiance;
			radiance.xyz = albedo_dec.xyz * ComputePointLightLambertian(CurrPointLight, voxel_center, albedo_dec.xyz, normal_dec.xyz);
			radiance.xyz = clamp(radiance.xyz, vec3(0.0), vec3(1.0));
			//radiance.w = 1.0;
			uint u32_radiance = ColorVec4ToUint(radiance);
			//sadly we need a lock here to avoid repeat injection
			uint old_val = imageLoad(TexVoxelRadiance, load_coord).x;
			if(((old_val & 0xFF000000) >> 24U) <= CurrLightPass - 1)
			{
				uvec4 new_v4;
				new_v4.x = (u32_radiance & 0xFF000000) >> 24U;
				new_v4.y = (u32_radiance & 0x00FF0000) >> 16U;
				new_v4.z = (u32_radiance & 0x0000FF00) >> 8U;
				new_v4.w = CurrLightPass;

				uvec4 old_v4;
				old_v4.x = (old_val & 0x000000FF);
				old_v4.y = (old_val & 0x0000FF00) >> 8U;
				old_v4.z = (old_val & 0x00FF0000) >> 16U;
				old_v4.w = 0;

				new_v4 = clamp(new_v4 + old_val, 0, 0xFF);
				uint new_val = (new_v4.x) | (new_v4.y << 8U) | (new_v4.z << 16U) | (new_v4.w << 24U);
				imageAtomicCompSwap(TexVoxelRadiance, load_coord, old_val, new_val);
			}
			//imageStore(TexVoxelRadiance, load_coord, uvec4(u32_radiance, 0, 0, 0));
	   }
	}
}