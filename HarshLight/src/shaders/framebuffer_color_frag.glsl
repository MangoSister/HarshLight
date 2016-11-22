#version 450 core

in vec2 vs_Texcoord;
out vec4 fragColor;

uniform sampler2D TexScreen;

void main()
{ 
    fragColor = texture(TexScreen, vs_Texcoord);
}