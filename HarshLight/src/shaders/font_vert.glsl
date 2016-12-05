#version 450 core
layout (location = 0) in vec4 app_Position;

out vec2 vs_TexCoords;

uniform mat4 ProjMtx;

void main()
{
    gl_Position = ProjMtx * vec4(app_Position.xy, 0.0, 1.0);
    vs_TexCoords = app_Position.zw;
}  