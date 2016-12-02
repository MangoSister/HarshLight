#version 450 core

in vec2 vs_Texcoord;
in vec3 vs_WorldPosition;
in vec3 vs_WorldNormal;
  
out vec4 fragColor;

uniform sampler2D TexAlbedo;

void main()
{
    fragColor = texture(TexAlbedo, vs_Texcoord);
	fragColor = vec4(vs_WorldNormal * 0.5 + 0.5, 1.0);
} 