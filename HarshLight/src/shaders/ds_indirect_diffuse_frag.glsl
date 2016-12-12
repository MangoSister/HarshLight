#version 450 core
layout (location = 0) out vec3 IndirectDiffuse;

#define SQRT3 1.73205080757
#define TAN30 0.57735026919
#define TAN10 0.1763269807
#define MAX_TRACE_DIST 200 //500, 1000?

in vec2 vs_Texcoord;

/* --------------  G-Buffer  ----------- */
uniform sampler2D GPositionAndSpecPower;
uniform sampler2D GNormalAndTangent;
uniform sampler2D GAlbedoAndSpecIntensity;
/* --------------------------------------- */

/* --------------  Camera Info  ----------- */
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
/* --------------------------------------- */

/* --------------  VCT Info  ----------- */
uniform float VoxelDim;
uniform float VoxelScale;

uniform sampler3D ImgRadianceLeaf;
uniform sampler3D ImgRadianceInterior[6];
/* --------------------------------------- */

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
	IndirectDiffuse.xyz = vec3(0.0, 0.0, 0.0);
	vec3 position = texture(GPositionAndSpecPower, vs_Texcoord).xyz;

	vec4 normal_tangent = texture(GNormalAndTangent, vs_Texcoord);
	vec3 normal;
	normal.xy = normal_tangent.xy;
	normal.z = sqrt(1.0 - dot(normal.xy, normal.xy));
	vec3 tangent;
	tangent.xy = normal_tangent.zw;
	tangent.z = sqrt(1.0 - dot(tangent.xy, tangent.xy));
	vec3 bitangent = cross(normal, tangent);

	//indirect diffuse
	IndirectDiffuse.xyz += 0.2 * VoxelConeTracing(position, normal, TAN30, MAX_TRACE_DIST);
	//cos(60) = 0.5
	IndirectDiffuse.xyz += 0.1 * VoxelConeTracing(position, normalize(normal + SQRT3 * tangent), TAN30, MAX_TRACE_DIST);
	IndirectDiffuse.xyz += 0.1 * VoxelConeTracing(position, normalize(normal - SQRT3 * tangent), TAN30, MAX_TRACE_DIST);
	IndirectDiffuse.xyz += 0.1 * VoxelConeTracing(position, normalize(normal + SQRT3 * bitangent), TAN30, MAX_TRACE_DIST);
	IndirectDiffuse.xyz += 0.1 * VoxelConeTracing(position, normalize(normal - SQRT3 * bitangent), TAN30, MAX_TRACE_DIST);
}
