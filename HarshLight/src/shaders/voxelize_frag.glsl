#version 450 core

in vec2 gs_Texcoord;
in vec3 gs_WorldPosition;
in vec3 gs_VoxelCoord;
in vec3 gs_WorldNormal;
in vec2 gs_ExpandedNDCPos;

in flat vec4 gs_BBox;
in flat vec3 gs_VoxelSpaceTri[3];
in flat ivec3 gs_ProjDir;

layout (std140, binding = 0) uniform MainCamMtx
{
    mat4 View;
    mat4 Proj;
	vec4 CamWorldPos;
};


uniform vec2 VoxelDim;

uniform sampler2D TexAlbedo;
layout (binding = 1, r32ui) coherent volatile uniform uimage3D TexVoxel;
layout (binding = 2, r32ui) coherent volatile uniform uimage3D TexVoxelNormal;

//dummy output
out vec4 fragColor;

bool TriangleVoxelTest(in vec3 tri_min, in vec3 tri_max, in vec3 tri_normal, in float tri_plane_d, in vec3 edge[3], in ivec3 voxel)
{
	//based on SAT (separate axis theorem), 13 axes in total

	//AABB
	vec3 voxel_min = vec3(voxel);// - vec3(0.5, 0.5, 0.5);
	vec3 voxel_max = vec3(voxel) + vec3(1.0, 1.0, 1.0);

	if( (tri_max.x < voxel_min.x) || (tri_min.x > voxel_max.x) ||
		(tri_max.y < voxel_min.y) || (tri_min.y > voxel_max.y) ||
		(tri_max.z < voxel_min.z) || (tri_min.z > voxel_max.z) )
		return false;
	

	// triangle normal (1)
	vec3 p = voxel_min;
	if (tri_normal.x >= 0.0)
		p.x = voxel_max.x;
	if (tri_normal.y >= 0.0)
		p.y = voxel_max.y;
	if (tri_normal.z >= 0.0)
		p.z = voxel_max.z;

	float p_dot = dot(p, tri_normal) + tri_plane_d;
	if(p_dot < 0.0)
		return false;

	vec3 n = voxel_max;
	if (tri_normal.x >= 0.0)
		n.x = voxel_min.x;
	if (tri_normal.y >= 0.0)
		n.y = voxel_min.y;
	if (tri_normal.z >= 0.0)
		n.z = voxel_min.z;
	
	float n_dot = dot(n, tri_normal) + tri_plane_d;
	if(n_dot * p_dot > 0.0)
		return false;

	//cross products of edges from tri + voxel (3 * 3 = 9)
	vec3 voxel_axis[3] = vec3[3]( vec3(1, 0, 0), vec3(0, 1, 0), vec3(0, 0, 1) ); 
	for(int i = 0; i < 3; i++)
	{
		for(int j = 0; j < 3; j++)
		{
			vec3 axis = normalize(cross(edge[i], voxel_axis[j]));
			float pt0 = dot(axis, gs_VoxelSpaceTri[0]);
			float pt1 = dot(axis, gs_VoxelSpaceTri[1]);
			float pt2 = dot(axis, gs_VoxelSpaceTri[2]);
			float max_pt = max(max(pt0, pt1), pt2);
			float min_pt = min(min(pt0, pt1), pt2);

			vec3 pa = voxel_min;
			if (axis.x >= 0.0)
				pa.x = voxel_max.x;
			if (axis.y >= 0.0)
				pa.y = voxel_max.y;
			if (axis.z >= 0.0)
				pa.z = voxel_max.z;

			vec3 na = voxel_max;
			if (axis.x >= 0.0)
				na.x = voxel_min.x;
			if (axis.y >= 0.0)
				na.y = voxel_min.y;
			if (axis.z >= 0.0)
				na.z = voxel_min.z;

			float pa_dot = dot(pa, axis);
			float na_dot = dot(na, axis);

			if(max_pt < na_dot || min_pt > pa_dot)
				return false;
		}
	}
	return true;
}


uint ColorVec4ToUint(vec4 val) 
{
	val.xyz *= 255.0;
	return	(uint(val.x) & 0x000000FF) << 24U | 
			(uint(val.y) & 0x000000FF) << 16U | 
			(uint(val.z) & 0x000000FF) << 8U | 
			(uint(val.w) & 0x000000FF);
}

vec4 ColorUintToVec4(uint val) 
{
	float r = float((val & 0xFF000000) >> 24U);
	float g = float((val & 0x00FF0000) >> 16U);
	float b = float((val & 0x0000FF00) >> 8U);
	float a = float((val & 0x000000FF));

	vec4 o = vec4(r, g, b, a);
	o.xyz /= 255.0;
	o.xyz = clamp(o.xyz, vec3(0.0, 0.0, 0.0), vec3(1.0, 1.0, 1.0));
	return o;
}


void AccumulateAlbedo
(layout(r32ui) coherent volatile uimage3D albedo, ivec3 coords, vec4 val) 
{
	uint new_val = ColorVec4ToUint(val);
	uint prev_val = 0; 
	uint curr_val;

	// Loop as long as destination value gets changed by other threads
	while ( (curr_val = imageAtomicCompSwap(albedo, coords, prev_val, new_val) ) != prev_val) 
	{
		prev_val = curr_val;
		vec4 rval = ColorUintToVec4(curr_val);
		if(rval.w >= 255.0) // we can at most count 255 frags in one voxel
			break;
		rval.xyz = (rval.xyz * rval.w); // Denormalize
		vec4 curr_val_vf = rval + val; // Add new value
		curr_val_vf.xyz /= (curr_val_vf.w); // Renormalize
		new_val = ColorVec4ToUint(curr_val_vf);
	}
}

void AccumulateNormal
(layout(r32ui) coherent volatile uimage3D normal, ivec3 coords, vec3 val)
{
	vec4 val_4 = vec4(val, 1.0); //[-1, 1]
	//[-1, 1] -> [0, 1]
	//val_4.xyz = 0.5 * (val_4.xyz + vec3(1.0));
	uint new_val = ColorVec4ToUint(val_4);
	uint prev_val = 0; 
	uint curr_val;

	// Loop as long as destination value gets changed by other threads
	while ( (curr_val = imageAtomicCompSwap(normal, coords, prev_val, new_val) ) != prev_val) 
	{
		prev_val = curr_val;
		vec4 rval = ColorUintToVec4(curr_val);
		if(rval.w >= 255.0) // we can at most count 255 frags in one voxel
			break;
		rval.xyz = (rval.xyz * rval.w); // Denormalize avg
		//[0, 1] -> [-1, 1]
		rval.xyz = (rval.xyz * 2.0) - vec3(1.0);
		vec4 curr_val_vf = rval + val_4; // Add new value
		//[-1, 1] -> [0, 1]
		curr_val_vf.xyz = 0.5 * (curr_val_vf.xyz + vec3(1.0));
		curr_val_vf.xyz /= (curr_val_vf.w); // Renormalize avg
		new_val = ColorVec4ToUint(curr_val_vf);
	}
}

void main()
{
	if ( (any(lessThan(gs_ExpandedNDCPos, gs_BBox.xy)) || any(greaterThan(gs_ExpandedNDCPos, gs_BBox.zw))) )
		discard;

	fragColor = vec4(texture(TexAlbedo, gs_Texcoord).xyz, 1.0);

	vec3 tri_min;
	tri_min.x = min(min(gs_VoxelSpaceTri[0].x, gs_VoxelSpaceTri[1].x), gs_VoxelSpaceTri[2].x);
	tri_min.y = min(min(gs_VoxelSpaceTri[0].y, gs_VoxelSpaceTri[1].y), gs_VoxelSpaceTri[2].y);
	tri_min.z = min(min(gs_VoxelSpaceTri[0].z, gs_VoxelSpaceTri[1].z), gs_VoxelSpaceTri[2].z);
	vec3 tri_max;
	tri_max.x = max(max(gs_VoxelSpaceTri[0].x, gs_VoxelSpaceTri[1].x), gs_VoxelSpaceTri[2].x);
	tri_max.y = max(max(gs_VoxelSpaceTri[0].y, gs_VoxelSpaceTri[1].y), gs_VoxelSpaceTri[2].y);
	tri_max.z = max(max(gs_VoxelSpaceTri[0].z, gs_VoxelSpaceTri[1].z), gs_VoxelSpaceTri[2].z);
	
	vec3 edge[3];
	edge[0] = gs_VoxelSpaceTri[1] - gs_VoxelSpaceTri[0];
	edge[1] = gs_VoxelSpaceTri[2] - gs_VoxelSpaceTri[1];
	edge[2] = gs_VoxelSpaceTri[0] - gs_VoxelSpaceTri[2];

	vec3 tri_normal = normalize(cross(edge[0].xyz, edge[2].xyz));
	float tri_plane_d = -dot(gs_VoxelSpaceTri[0], tri_normal);

	uint u32_fragColor = ColorVec4ToUint(fragColor);

	ivec3 next = ivec3(gs_VoxelCoord);
	if( all(greaterThanEqual(next, ivec3(0,0,0))) && all(lessThan(next, ivec3(VoxelDim.xxx)) ) )
	{	
		if( TriangleVoxelTest(tri_min, tri_max, tri_normal, tri_plane_d, edge, next) )
		{
			AccumulateAlbedo(TexVoxel, next, fragColor);
			AccumulateNormal(TexVoxelNormal, next, gs_WorldNormal);
			//imageStore(TexVoxel[0], next, uvec4(u32_fragColor, 0xFF00FFFF, 0xFF00FFFF, 0xFF00FFFF));
		}
	}


	next = ivec3(gs_VoxelCoord) + gs_ProjDir;
	if( all(greaterThanEqual(next, ivec3(0,0,0))) && all(lessThan(next, ivec3(VoxelDim.xxx)) ) )
	{
		if( TriangleVoxelTest(tri_min, tri_max, tri_normal, tri_plane_d, edge, ivec3(gs_VoxelCoord) + gs_ProjDir) )
		{
			AccumulateAlbedo(TexVoxel, next, fragColor);
			AccumulateNormal(TexVoxelNormal, next, gs_WorldNormal);
			//imageStore(TexVoxel[0], next, uvec4(u32_fragColor, 0xFF00FFFF, 0xFF00FFFF, 0xFF00FFFF));
		}
	}

	next = ivec3(gs_VoxelCoord) - gs_ProjDir;
	if( all(greaterThanEqual(next, ivec3(0,0,0))) && all(lessThan(next, ivec3(VoxelDim.xxx)) ) )
	{
		if( TriangleVoxelTest(tri_min, tri_max, tri_normal, tri_plane_d, edge, ivec3(gs_VoxelCoord) - gs_ProjDir) )
		{
			AccumulateAlbedo(TexVoxel, next, fragColor);
			AccumulateNormal(TexVoxelNormal, next, gs_WorldNormal);
			//imageStore(TexVoxel[0], next, uvec4(u32_fragColor, 0xFF00FFFF, 0xFF00FFFF, 0xFF00FFFF));
		}
	}

}