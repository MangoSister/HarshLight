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
};

uniform vec2 VoxelDim;

uniform sampler2D TexAlbedo;
layout (binding = 1, rgba8) uniform image3D TexVoxel;

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






void main()
{
	if ( (any(lessThan(gs_ExpandedNDCPos, gs_BBox.xy)) || any(greaterThan(gs_ExpandedNDCPos, gs_BBox.zw))) )
		discard;

	fragColor = texture(TexAlbedo, gs_Texcoord);

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


	if( TriangleVoxelTest(tri_min, tri_max, tri_normal, tri_plane_d, edge, ivec3(gs_VoxelCoord)) )
		imageStore(TexVoxel, ivec3(gs_VoxelCoord), fragColor);
	
	if( TriangleVoxelTest(tri_min, tri_max, tri_normal, tri_plane_d, edge, ivec3(gs_VoxelCoord) + gs_ProjDir) )
		imageStore(TexVoxel, ivec3(gs_VoxelCoord) + gs_ProjDir, fragColor);
	
	if( TriangleVoxelTest(tri_min, tri_max, tri_normal, tri_plane_d, edge, ivec3(gs_VoxelCoord) - gs_ProjDir) )
		imageStore(TexVoxel, ivec3(gs_VoxelCoord) - gs_ProjDir, fragColor);
	//imageStore(TexVoxel, ivec3(0, 0, 0), vec4(0.5,1,0.5,1));
}