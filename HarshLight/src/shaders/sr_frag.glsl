#version 450 core

in vec2 vs_Texcoord;
in vec3 vs_WorldPosition;
in vec3 vs_WorldNormal;

out vec4 fragColor;

uniform sampler2D TexAlbedo;

void main()
{
    fragColor = vec4(0, 0, 1, 0.5);
} 