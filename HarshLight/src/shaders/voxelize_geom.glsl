#version 450 core

layout(triangles, invocations = 1) in;
layout(triangle_strip, max_vertices = 3) out;

layout (std140, binding = 0) uniform MainCamMtx
{
    mat4 View;
    mat4 Proj;
};

uniform vec2 VoxelDim;
uniform mat4 ViewMtxToDown;
uniform mat4 ViewMtxToLeft;
uniform mat4 ViewMtxToForward;
//uniform float PixelDiagonal;

in vec2 vs_Texcoord[];
in vec3 vs_WorldPosition[];
in vec3 vs_WorldNormal[];

out vec2 gs_Texcoord;
out vec3 gs_WorldPosition;
out vec3 gs_VoxelCoord;
out vec3 gs_WorldNormal;
out vec2 gs_ExpandedNDCPos;

out flat vec4 gs_BBox;

void ExpandTri(inout vec4 screen_pos[3])
{
	vec3 plane[3];

	const vec3 e01 = screen_pos[1].xyw - screen_pos[0].xyw;
	const vec3 e12 = screen_pos[2].xyw - screen_pos[1].xyw;

	plane[0] = cross(e01, screen_pos[0].xyw);
	plane[1] = cross(e12, screen_pos[1].xyw);
	plane[2] = cross(screen_pos[0].xyw - screen_pos[2].xyw, screen_pos[2].xyw);

	const vec2 hPixel = 2.0 * vec2(1.0, 1.0) / VoxelDim;

	// flip if not CCW
	const float ccw = sign(cross(e01, e12).z);
	
	plane[0].z -= dot(hPixel.xy, abs(plane[0].xy)) * ccw;
	plane[1].z -= dot(hPixel.xy, abs(plane[1].xy)) * ccw;
	plane[2].z -= dot(hPixel.xy, abs(plane[2].xy)) * ccw;

	vec3 new_pos[3];
	new_pos[0] = cross(plane[2], plane[0]);
	new_pos[1] = cross(plane[0], plane[1]);
	new_pos[2] = cross(plane[1], plane[2]);
	
	new_pos[0] /= new_pos[0].z;
	new_pos[1] /= new_pos[1].z;
	new_pos[2] /= new_pos[2].z;

	screen_pos[0].xy = new_pos[0].xy;
	screen_pos[1].xy = new_pos[1].xy;
	screen_pos[2].xy = new_pos[2].xy;
}


void main()
{
	//view space face normal
	const vec3 world_e01 = vs_WorldPosition[1] - vs_WorldPosition[0];
	const vec3 world_e02 = vs_WorldPosition[2] - vs_WorldPosition[0];
	const vec3 world_face_normal = normalize(cross(world_e01, world_e02));
	const vec3 abs_world_face_normal = abs(world_face_normal);
	float dominant_axis = max(abs_world_face_normal.x, max(abs_world_face_normal.y, abs_world_face_normal.z));
	mat4 swizzle_view_mtx = ViewMtxToDown;

	if(dominant_axis == abs_world_face_normal.x)
	{
		swizzle_view_mtx = ViewMtxToLeft;
	}
	else if(dominant_axis == abs_world_face_normal.z)
	{
		swizzle_view_mtx = ViewMtxToForward;
	}

	const mat4 proj_swizzle_view = Proj * swizzle_view_mtx;
	const mat4 inv_proj_swizzle_view = inverse(proj_swizzle_view);
	const mat4 it_proj_swizzle_view = transpose(inv_proj_swizzle_view);
	vec4 screen_pos[3];
	//ONLY CORRECT FOR ORTHOGONAL PROJECTION!!
	//do perspective division here
	//ensure all w components to be 1
	screen_pos[0] = proj_swizzle_view * vec4(vs_WorldPosition[0], 1.0);
	screen_pos[1] = proj_swizzle_view * vec4(vs_WorldPosition[1], 1.0);
	screen_pos[2] = proj_swizzle_view * vec4(vs_WorldPosition[2], 1.0);
	//screen_pos[0] /= screen_pos[0].w;
	//screen_pos[1] /= screen_pos[1].w;
	//screen_pos[2] /= screen_pos[2].w;
	//screen_pos are in NDC now

	gs_BBox.x = min(screen_pos[0].x, min(screen_pos[1].x, screen_pos[2].x));
	gs_BBox.y = min(screen_pos[0].y, min(screen_pos[1].y, screen_pos[2].y));
	gs_BBox.z = max(screen_pos[0].x, max(screen_pos[1].x, screen_pos[2].x));
	gs_BBox.w = max(screen_pos[0].y, max(screen_pos[1].y, screen_pos[2].y));

	const vec2 padding = 2.0 * vec2(1.0, 1.0) / VoxelDim;
	gs_BBox.xy -= padding;
	gs_BBox.zw += padding;

	vec3 proj_normal0 = normalize((it_proj_swizzle_view * vec4(world_face_normal, 0)).xyz);
	float d0 = dot(screen_pos[0].xyz, proj_normal0);

	//screen_pos should be in CCW order, if not then flipped in the function
	ExpandTri(screen_pos);
	//screen_pos z components should remain unchanged
	
	screen_pos[0].z = (d0 - dot(screen_pos[0].xy, proj_normal0.xy) ) / proj_normal0.z;
	screen_pos[1].z = (d0 - dot(screen_pos[1].xy, proj_normal0.xy) ) / proj_normal0.z;
	screen_pos[2].z = (d0 - dot(screen_pos[2].xy, proj_normal0.xy) ) / proj_normal0.z;

	gs_VoxelCoord.xyz = (Proj * View * inv_proj_swizzle_view * screen_pos[0]).xyz;
	gs_VoxelCoord.xyz = VoxelDim.xxx * 0.5 * (gs_VoxelCoord.xyz + vec3(1.0, 1.0, 1.0));
	//gs_VoxelCoord.xyz = gl_in[0].gl_Position.xyz;
	gs_Texcoord = vs_Texcoord[0];
	gs_WorldPosition = vs_WorldPosition[0];
	gs_WorldNormal = vs_WorldNormal[0];
	gs_ExpandedNDCPos = screen_pos[0].xy;
	gl_Position = screen_pos[0];
	//gl_Position.z =  (-screen_pos[0].x * proj_normal0.x - screen_pos[0].y * proj_normal0.y + d0) / proj_normal0.z;
	EmitVertex();

	gs_VoxelCoord.xyz = (Proj * View * inv_proj_swizzle_view * screen_pos[1]).xyz;
	gs_VoxelCoord.xyz = VoxelDim.xxx * 0.5 * (gs_VoxelCoord.xyz + vec3(1.0, 1.0, 1.0));
	//gs_VoxelCoord.xyz = gl_in[1].gl_Position.xyz;
	gs_Texcoord = vs_Texcoord[1];
	gs_WorldPosition = vs_WorldPosition[1];
	gs_WorldNormal = vs_WorldNormal[1];
	gs_ExpandedNDCPos = screen_pos[1].xy;
	gl_Position = screen_pos[1];
	//gl_Position.z =  (-screen_pos[1].x * proj_normal0.x - screen_pos[1].y * proj_normal0.y + d0) / proj_normal0.z;
	EmitVertex();

	gs_VoxelCoord.xyz = (Proj * View * inv_proj_swizzle_view * screen_pos[2]).xyz;
	gs_VoxelCoord.xyz = VoxelDim.xxx * 0.5 * (gs_VoxelCoord.xyz + vec3(1.0, 1.0, 1.0));
	//gs_VoxelCoord.xyz = gl_in[2].gl_Position.xyz;
	gs_Texcoord = vs_Texcoord[2];
	gs_WorldPosition = vs_WorldPosition[2];
	gs_WorldNormal = vs_WorldNormal[2];
	gs_ExpandedNDCPos = screen_pos[2].xy;
	gl_Position = screen_pos[2];
	//gl_Position.z =  (-screen_pos[2].x * proj_normal0.x - screen_pos[2].y * proj_normal0.y + d0) / proj_normal0.z;
	EmitVertex();

	EndPrimitive();
}
