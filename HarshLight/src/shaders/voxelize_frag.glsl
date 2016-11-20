#version 330 core
in vec3 worldNormal;
in vec2 texcoord;

uniform sampler2D TexAlbedo;

out vec4 fragColor;

void main()
{
	fragColor = texture(TexAlbedo, texcoord);
    //fragColor = vec4(worldNormal, 1);
} 