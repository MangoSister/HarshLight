#version 450 core

layout(triangles, invocations = 1) in;
layout(triangle_strip, max_vertices = 3) out;

layout (std140, binding = 0) uniform MainCamMtx
{
    mat4 View;
    mat4 Proj;
};

//uniform float PixelDiagonal;

in vec2 vs_Texcoord[];
in vec3 vs_WorldPosition[];
in vec3 vs_WorldNormal[];

out vec2 gs_Texcoord;
out vec3 gs_WorldPosition;
out float gs_NDCLinearDepth;
out vec3 gs_WorldNormal;
out vec4 gs_BBox;
out flat ivec3 gs_ProjDir;

void expandTriangle(inout vec4 screenPos[3])
{
	vec2 edge[3];
	edge[0] = screenPos[1].xy - screenPos[0].xy;
	edge[1] = screenPos[2].xy - screenPos[1].xy;
	edge[2] = screenPos[0].xy - screenPos[2].xy;

	vec2 edgeNormal[3];
	edgeNormal[0] = normalize(edge[0]);
	edgeNormal[1] = normalize(edge[1]);
	edgeNormal[2] = normalize(edge[2]);
	edgeNormal[0] = vec2(-edgeNormal[0].y, edgeNormal[0].x);
	edgeNormal[1] = vec2(-edgeNormal[1].y, edgeNormal[1].x);
	edgeNormal[2] = vec2(-edgeNormal[2].y, edgeNormal[2].x);

    // If triangle is back facing, flip it's edge normals so triangle does not shrink.
    vec3 a = normalize(screenPos[1].xyz - screenPos[0].xyz);
	vec3 b = normalize(screenPos[2].xyz - screenPos[0].xyz);
	vec3 clipSpaceNormal = cross(a, b);
    if (clipSpaceNormal.z < 0.0)
    {
        edgeNormal[0] *= -1.0;
        edgeNormal[1] *= -1.0;
        edgeNormal[2] *= -1.0;
    }

	vec3 edgeDist;
	edgeDist.x = dot(edgeNormal[0], screenPos[0].xy);
	edgeDist.y = dot(edgeNormal[1], screenPos[1].xy);
	edgeDist.z = dot(edgeNormal[2], screenPos[2].xy);

	float PixelDiagonal = 1.414 / 256.0;

	screenPos[0].xy = screenPos[0].xy - PixelDiagonal * (edge[2] / dot(edge[2], edgeNormal[0]) + edge[0] / dot(edge[0], edgeNormal[2]));
	screenPos[1].xy = screenPos[1].xy - PixelDiagonal * (edge[0] / dot(edge[0], edgeNormal[1]) + edge[1] / dot(edge[1], edgeNormal[0]));
	screenPos[2].xy = screenPos[2].xy - PixelDiagonal * (edge[1] / dot(edge[1], edgeNormal[2]) + edge[2] / dot(edge[2], edgeNormal[1]));
}


void main()
{
	//view space face normal
	vec3 view_e01 = gl_in[1].gl_Position.xyz - gl_in[0].gl_Position.xyz;
	vec3 view_e02 = gl_in[2].gl_Position.xyz - gl_in[0].gl_Position.xyz;
	vec3 view_normal = abs(cross(view_e01, view_e02));
	float dominant_axis = max(view_normal.x, max(view_normal.y, view_normal.z));
	if(dominant_axis == view_normal.x)
	{
		gs_ProjDir = ivec3(2, 1, 0);
	}
	else if(dominant_axis == view_normal.y)
	{
		gs_ProjDir = ivec3(0, 2, 1);
	}
	else
	{
		gs_ProjDir = ivec3(0, 1, 2);
	}

	const mat3 identity = mat3(vec3(1.0, 0.0, 0.0), vec3(0.0, 1.0, 0.0), vec3(0.0, 0.0, 1.0));
	mat3 swizzle = mat3(identity[gs_ProjDir[0]], identity[gs_ProjDir[1]], identity[gs_ProjDir[2]]);

	vec4 screen_pos[3];
	screen_pos[0] = Proj * mat4(swizzle) * gl_in[0].gl_Position;
	screen_pos[1] = Proj * mat4(swizzle) * gl_in[1].gl_Position;
	screen_pos[2] = Proj * mat4(swizzle) * gl_in[2].gl_Position;
	screen_pos[0] /= screen_pos[0].w;
	screen_pos[1] /= screen_pos[1].w;
	screen_pos[2] /= screen_pos[2].w;
	
	expandTriangle(screen_pos);

	gs_NDCLinearDepth = screen_pos[0].z;
	gs_Texcoord = vs_Texcoord[0];
	gs_WorldPosition = vs_WorldPosition[0];
	gs_WorldNormal = vs_WorldNormal[0];
	gl_Position = screen_pos[0];
	EmitVertex();

	gs_NDCLinearDepth = screen_pos[1].z;
	gs_Texcoord = vs_Texcoord[1];
	gs_WorldPosition = vs_WorldPosition[1];
	gs_WorldNormal = vs_WorldNormal[1];
	gl_Position = screen_pos[1];
	EmitVertex();

	gs_NDCLinearDepth = screen_pos[2].z;
	gs_Texcoord = vs_Texcoord[2];
	gs_WorldPosition = vs_WorldPosition[2];
	gs_WorldNormal = vs_WorldNormal[2];
	gl_Position = screen_pos[2];
	EmitVertex();

	EndPrimitive();
}
