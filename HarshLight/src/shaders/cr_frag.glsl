#version 450 core

in vec2 gs_Texcoord;
in vec3 gs_WorldPosition;
in vec3 gs_WorldNormal;
in noperspective vec3 gs_NDCPos;
in vec4 gs_BBox;

out vec4 fragColor;

uniform sampler2D TexAlbedo;

void main()
{
	if ( (all(greaterThanEqual(gs_NDCPos.xy, gs_BBox.xy)) && all(lessThanEqual(gs_NDCPos.xy, gs_BBox.zw))) )
		fragColor = vec4(1, 0, 0, 0.5);
	else discard;
    
} 