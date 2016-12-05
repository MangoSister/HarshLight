#version 450 core

in vec2 vs_Texcoord;
out vec4 fragColor;

uniform samplerCube TexCube;

uniform int Face;

void main()
{ 
	if(Face == 0)
		//fragColor.xyz = texture(TexCube, vec3(1.0, 0.0, 0.0)).xxx;
		fragColor.xyz = texture(TexCube, vec3(1.0, vs_Texcoord * 2.0 - vec2(1.0))).xxx;
	else if(Face == 1)
		fragColor.xyz = texture(TexCube, vec3(-1.0, vs_Texcoord * 2.0 - vec2(1.0))).xxx;
	else if(Face == 2)
		fragColor.xyz = texture(TexCube, vec3(vs_Texcoord.x * 2.0 - 1.0, 1.0, vs_Texcoord.y * 2.0 - 1.0)).xxx;
	else if(Face == 3)
		fragColor.xyz = texture(TexCube, vec3(vs_Texcoord.x * 2.0 - 1.0, -1.0, vs_Texcoord.y * 2.0 - 1.0)).xxx;
	else if(Face == 4)
		fragColor.xyz = texture(TexCube, vec3(vs_Texcoord * 2.0 - vec2(1.0), 1.0)).xxx;
	else
		fragColor.xyz = texture(TexCube, vec3(vs_Texcoord * 2.0 - vec2(1.0), -1.0)).xxx;

	fragColor.w = 0.5;
}