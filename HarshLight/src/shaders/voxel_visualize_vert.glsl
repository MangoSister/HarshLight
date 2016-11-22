#version 450 core

layout (location = 0) in vec3 app_Position;
layout (location = 1) in vec3 app_Normal;
layout (location = 2) in vec2 app_Uv;

out vec2 vs_Texcoord;
out vec3 vs_WorldPosition;
out vec3 vs_WorldNormal;

layout (std140) uniform CamMtx
{
    mat4 View;
    mat4 Proj;
};

uniform mat4 Model;

void main()
{
    gl_Position =  Proj * View * Model * vec4(app_Position, 1.0);
	vs_Texcoord = app_Uv;
	vs_WorldPosition = vec3(Model * vec4(app_Position, 1.0));
	vs_WorldNormal = normalize(mat3(Model) * app_Normal);
}