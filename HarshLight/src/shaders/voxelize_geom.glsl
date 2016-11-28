#version 450 core

layout(triangles, invocations = 1) in;
layout(triangle_strip, max_vertices = 3) out;

layout (std140, binding = 0) uniform MainCamMtx
{
    mat4 View;
    mat4 Proj;
};

uniform vec2 VoxelDim;
//uniform float PixelDiagonal;

in vec2 vs_Texcoord[];
in vec3 vs_WorldPosition[];
in vec3 vs_WorldNormal[];

out vec2 gs_Texcoord;
out vec3 gs_WorldPosition;
out vec3 gs_ViewPosition;
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
	//view space face normal
	vec3 view_e01 = gl_in[1].gl_Position.xyz - gl_in[0].gl_Position.xyz;
	vec3 view_e02 = gl_in[2].gl_Position.xyz - gl_in[0].gl_Position.xyz;
	vec3 view_normal = abs(cross(view_e01, view_e02));
	float dominant_axis = max(view_normal.x, max(view_normal.y, view_normal.z));
	ivec3 proj_dir;
	mat4 swizzleMatrix;
	if(dominant_axis == view_normal.x)
	{
		proj_dir = ivec3(2, 1, 0);
		swizzleMatrix = mat4(vec4(0.0, 0.0, 1.0, 0.0),
							 vec4(0.0, 1.0, 0.0, 0.0),
							 vec4(1.0, 0.0, 0.0, 0.0),
							 vec4(0.0, 0.0, 0.0, 1.0));
	}
	else if(dominant_axis == view_normal.y)
	{
		proj_dir = ivec3(0, 2, 1);
		swizzleMatrix = mat4(vec4(1.0, 0.0, 0.0, 0.0),
						 	 vec4(0.0, 0.0, 1.0, 0.0),
							 vec4(0.0, 1.0, 0.0, 0.0),
							 vec4(0.0, 0.0, 0.0, 1.0));
	}
	else
	{
		proj_dir = ivec3(0, 1, 2);
		swizzleMatrix = mat4(vec4(1.0, 0.0, 0.0, 0.0),
							 vec4(0.0, 1.0, 0.0, 0.0),
							 vec4(0.0, 0.0, 1.0, 0.0),
							 vec4(0.0, 0.0, 0.0, 1.0));
	}

	const mat3 identity = mat3(vec3(1.0, 0.0, 0.0), vec3(0.0, 1.0, 0.0), vec3(0.0, 0.0, 1.0));
	mat3 swizzle = mat3(identity[proj_dir.x], identity[proj_dir.y], identity[proj_dir.z]);

	vec4 screen_pos[3];
	//ONLY CORRECT FOR ORTHOGONAL PROJECTION!!
	//do perspective division here
	//ensure all w components to be 1
	screen_pos[0] = Proj * swizzleMatrix * gl_in[0].gl_Position;
	screen_pos[1] = Proj * swizzleMatrix * gl_in[1].gl_Position;
	screen_pos[2] = Proj * swizzleMatrix * gl_in[2].gl_Position;
	screen_pos[0] /= screen_pos[0].w;
	screen_pos[1] /= screen_pos[1].w;
	screen_pos[2] /= screen_pos[2].w;
	//screen_pos are in NDC now

	gs_BBox.x = min(screen_pos[0].x, min(screen_pos[1].x, screen_pos[2].x));
	gs_BBox.y = min(screen_pos[0].y, min(screen_pos[1].y, screen_pos[2].y));
	gs_BBox.z = max(screen_pos[0].x, max(screen_pos[1].x, screen_pos[2].x));
	gs_BBox.w = max(screen_pos[0].y, max(screen_pos[1].y, screen_pos[2].y));

	const vec2 padding = vec2(1.0, 1.0) / VoxelDim;
	gs_BBox.xy -= padding;
	gs_BBox.zw += padding;

	//screen_pos should be in CCW order, if not then flipped in the function
	ExpandTri(screen_pos);
	//screen_pos z components should remain unchanged

	gs_ViewPosition.xyz = gl_in[0].gl_Position.xyz;
	gs_Texcoord = vs_Texcoord[0];
	gs_WorldPosition = vs_WorldPosition[0];
	gs_WorldNormal = vs_WorldNormal[0];
	gs_ExpandedNDCPos = screen_pos[0].xy;
	gl_Position = screen_pos[0];
	EmitVertex();

	gs_ViewPosition.xyz = gl_in[1].gl_Position.xyz;
	gs_Texcoord = vs_Texcoord[1];
	gs_WorldPosition = vs_WorldPosition[1];
	gs_WorldNormal = vs_WorldNormal[1];
	gs_ExpandedNDCPos = screen_pos[1].xy;
	gl_Position = screen_pos[1];
	EmitVertex();

	gs_ViewPosition.xyz = gl_in[2].gl_Position.xyz;
	gs_Texcoord = vs_Texcoord[2];
	gs_WorldPosition = vs_WorldPosition[2];
	gs_WorldNormal = vs_WorldNormal[2];
	gs_ExpandedNDCPos = screen_pos[2].xy;
	gl_Position = screen_pos[2];
	EmitVertex();

	EndPrimitive();
}
