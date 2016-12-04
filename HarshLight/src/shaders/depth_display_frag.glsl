#version 450 core

in vec2 vs_Texcoord;
out vec4 fragColor;

uniform sampler2D TexDepth;

void main()
{ 
    fragColor.xyz = texture(TexDepth, vs_Texcoord).xxx;
	fragColor.w = 0.5;
}