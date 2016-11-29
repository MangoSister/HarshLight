#version 450 core

in vec2 gs_Texcoord;
in vec3 gs_WorldPosition;
in vec3 gs_VoxelCoord;
in vec3 gs_WorldNormal;
in vec2 gs_ExpandedNDCPos;

in flat vec4 gs_BBox;

layout (std140, binding = 0) uniform MainCamMtx
{
    mat4 View;
    mat4 Proj;
};

uniform vec2 VoxelDim;

uniform sampler2D TexAlbedo;
layout (binding = 1, rgba8) uniform image3D TexVoxel;

//no output
out vec4 fragColor;

void main()
{
	if ( (any(lessThan(gs_ExpandedNDCPos, gs_BBox.xy)) || any(greaterThan(gs_ExpandedNDCPos, gs_BBox.zw))) )
		discard;

	fragColor = texture(TexAlbedo, gs_Texcoord);

	//vec3 coords = (Proj * vec4(gs_ViewPosition, 1.0)).xyz;
	//coords = VoxelDim.xxx * 0.5 * (coords + vec3(1.0, 1.0, 1.0));

	//for(int i = -1; i <= 1; i++)
	//	for(int j = -1; j <= 1; j++)
	//		for(int k = -1; k <= 1; k++)
	//imageStore(TexVoxel, ivec3(floor(coords)) + ivec3(i,j,k), fragColor);
	
	imageStore(TexVoxel, ivec3(gs_VoxelCoord), fragColor);

	//imageStore(TexVoxel, ivec3(0, 0, 0), vec4(0.5,1,0.5,1));
}