#version 450 core

layout(triangles, invocations = 1) in;
layout(triangle_strip, max_vertices = 18) out;

layout (std140, binding = 3) uniform LightCapture
{
	mat4 CubeLightMtx[6];
	vec4 PointLightWorldPos;
	vec2 CaptureRange;
};

out vec4 gs_WorldPosition;

void main()
{
	for(int face = 0; face < 6; face++)
	{
		gl_Layer = face; // built-in variable that specifies to which face we render.
		for(int i = 0; i < 3; i++) // for each triangle's vertices
        {
            gs_WorldPosition = gl_in[i].gl_Position;
            gl_Position = CubeLightMtx[face] * gs_WorldPosition;
            EmitVertex();
        }    
        EndPrimitive();
	}
}