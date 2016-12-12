#version 450 core

layout (location = 0) in vec3 app_Position;
layout (location = 1) in vec3 app_Normal;
layout (location = 2) in vec2 app_Uv;

out vec2 vs_Texcoord;

uniform mat4 Model;

void main()
{
    gl_Position = Model * vec4(app_Position.x, app_Position.y, 0.0f, 1.0f); 
    vs_Texcoord = app_Uv;
}  