#version 450 core

#define LOCAL_SIZE 8
layout(local_size_x = LOCAL_SIZE, local_size_y = LOCAL_SIZE) in;

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

layout (binding = 0) uniform sampler2D depthMap;
layout (binding = 1, r32ui) coherent volatile uniform uimage3D TexVoxelAlbedo;
layout (binding = 2, r32ui) coherent volatile uniform uimage3D TexVoxelNormal;
layout (binding = 3, r32ui) coherent volatile uniform uimage3D TexVoxelRadiance;

uniform mat4 LightProjMtx;
uniform DirLight CurrDirLight;

layout (std140, binding = 1) uniform VoxelCamMtx
{
    mat4 CamVoxelViewMtx;
    mat4 CamVoxelProjMtx;
	vec4 CamVoxelWorldPos;
};

vec3 ComputeDirLightLambertian(DirLight light, vec3 world_pos, vec3 albedo, vec3 normal)
{
	vec3 light_dir = normalize(-light.direction.xyz);
	float diffuse_intensity = max(dot(normalize(normal), light_dir), 0.0);

	return vec3(diffuse_intensity) * light.color.xyz * light.color.w;
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
	uvec2 map_dim = gl_NumWorkGroups.xy * uvec2(gl_WorkGroupSize.x, gl_WorkGroupSize.y);
	vec3 voxel_coord;
	voxel_coord.x = float(gl_GlobalInvocationID.x) / float(map_dim.x);
	voxel_coord.y = float(gl_GlobalInvocationID.y) / float(map_dim.y);
	voxel_coord.z = texture(depthMap, voxel_coord.xy).r;
	//now [0, 1]^3

	voxel_coord = 2.0 * voxel_coord - vec3(1.0);
	//now [-1, 1]^3
	
	voxel_coord = (inverse(LightProjMtx * CurrDirLight.lightMtx) * vec4(voxel_coord, 1.0)).xyz;
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
		radiance.xyz = albedo_dec.xyz * ComputeDirLightLambertian(CurrDirLight, voxel_center, albedo_dec.xyz, normal_dec.xyz);
		radiance.xyz = clamp(radiance.xyz, vec3(0.0), vec3(1.0));
		radiance.w = 1.0;
		uint u32_radiance = ColorVec4ToUint(radiance);
		imageStore(TexVoxelRadiance, load_coord, uvec4(u32_radiance, 0, 0, 0));
	}
}