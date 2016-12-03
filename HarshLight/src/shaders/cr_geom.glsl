#version 450 core

layout(triangles, invocations = 1) in;
layout(triangle_strip, max_vertices = 3) out;

layout (std140, binding = 0) uniform MainCamMtx
{
    mat4 View;
    mat4 Proj;
	vec4 CamWorldPos;
};

uniform vec2 VoxelDim;

in vec2 vs_Texcoord[];
in vec3 vs_WorldPosition[];
in vec3 vs_WorldNormal[];

out vec2 gs_Texcoord;
out vec3 gs_WorldPosition;
out noperspective vec3 gs_NDCPos;
out vec3 gs_WorldNormal;
out vec4 gs_BBox;
out flat ivec3 gs_ProjDir;


void ExpandTri(inout vec4 screen_pos[3])
{
	vec3 plane[3];

	const vec3 e01 = screen_pos[1].xyw - screen_pos[0].xyw;
	const vec3 e12 = screen_pos[2].xyw - screen_pos[1].xyw;

	plane[0] = cross(e01, screen_pos[0].xyw);
	plane[1] = cross(e12, screen_pos[1].xyw);
	plane[2] = cross(screen_pos[0].xyw - screen_pos[2].xyw, screen_pos[2].xyw);

	const vec2 hPixel = vec2(1.0, 1.0) / VoxelDim;

	// flip if not CCW
	const float ccw = sign(cross(e01, e12).z);
	
	plane[0].z -= dot(hPixel.xy, abs(plane[0].xy)) * ccw;
	plane[1].z -= dot(hPixel.xy, abs(plane[1].xy)) * ccw;
	plane[2].z -= dot(hPixel.xy, abs(plane[2].xy)) * ccw;

	vec3 new_pos[3];
	new_pos[0] = cross(plane[2], plane[0]);
	new_pos[1] = cross(plane[0], plane[1]);
	new_pos[2] = cross(plane[1], plane[2]);

	screen_pos[0].xy = new_pos[0].xy / new_pos[0].z;
	screen_pos[1].xy = new_pos[1].xy / new_pos[1].z;
	screen_pos[2].xy = new_pos[2].xy / new_pos[2].z;
}

void main()
{
	//ONLY CORRECT FOR ORTHOGONAL PROJECTION!!
	vec4 screen_pos[3];
	//do perspective division here
	//ensure all w components to be 1	
	screen_pos[0] = gl_in[0].gl_Position / gl_in[0].gl_Position.w;
	screen_pos[1] = gl_in[1].gl_Position / gl_in[1].gl_Position.w;
	screen_pos[2] = gl_in[2].gl_Position / gl_in[2].gl_Position.w;
	//screen_pos are in NDC now

	gs_BBox.x = min(screen_pos[0].x, min(screen_pos[1].x, screen_pos[2].x));
	gs_BBox.y = min(screen_pos[0].y, min(screen_pos[1].y, screen_pos[2].y));
	gs_BBox.z = max(screen_pos[0].x, max(screen_pos[1].x, screen_pos[2].x));
	gs_BBox.w = max(screen_pos[0].y, max(screen_pos[1].y, screen_pos[2].y));

	const vec2 padding = vec2(1.0, 1.0) / VoxelDim;
	gs_BBox.xy -= padding;
	gs_BBox.zw += padding;

	//screen_pos should be in CCW order
	ExpandTri(screen_pos);
	//screen_pos z components should remain unchanged

	gs_NDCPos.xyz = screen_pos[0].xyz;
	gs_Texcoord = vs_Texcoord[0];
	gs_WorldPosition = vs_WorldPosition[0];
	gs_WorldNormal = vs_WorldNormal[0];
	gl_Position = screen_pos[0];
	//gl_Position.w = 1;
	EmitVertex();

	gs_NDCPos.xyz = screen_pos[1].xyz;
	gs_Texcoord = vs_Texcoord[1];
	gs_WorldPosition = vs_WorldPosition[1];
	gs_WorldNormal = vs_WorldNormal[1];
	gl_Position = screen_pos[1];
	//gl_Position.w = 1;
	EmitVertex();

	gs_NDCPos.xyz = screen_pos[2].xyz;
	gs_Texcoord = vs_Texcoord[2];
	gs_WorldPosition = vs_WorldPosition[2];
	gs_WorldNormal = vs_WorldNormal[2];
	gl_Position = screen_pos[2];
	//gl_Position.w = 1;
	EmitVertex();

	EndPrimitive();
}