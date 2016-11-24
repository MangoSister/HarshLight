#version 450 core

in vec2 gs_Texcoord;
in vec3 gs_WorldPosition;
in vec3 gs_WorldNormal;
in vec4 gs_BBox;
in flat ivec3 gs_ProjDir;

uniform sampler2D TexAlbedo;
layout (binding = 1, rgba8) coherent uniform image3D TexVoxel;

//no output
out vec4 fragColor;

void main()
{
	//random test
    //   vec4 c = texture(TexAlbedo, gs_Texcoord);
	   //imageStore(TexVoxel, ivec3(0,0,0), vec4(1, 0, 0, 1));
	   //fragColor = c;

	//assume viewportSize.xy = voxelization dim
    ivec2 viewportSize = imageSize(TexVoxel).xy;
	vec2 bboxMin = floor((gs_BBox.xy * 0.5 + 0.5) * viewportSize);
	vec2 bboxMax = ceil((gs_BBox.zw * 0.5 + 0.5) * viewportSize);
    fragColor = texture(TexAlbedo, gs_Texcoord);

	const mat3 identity = mat3(vec3(1.0, 0.0, 0.0), vec3(0.0, 1.0, 0.0), vec3(0.0, 0.0, 1.0));
	mat3 swizzle = mat3(identity[gs_ProjDir[0]], identity[gs_ProjDir[1]], identity[gs_ProjDir[2]]);
    vec3 coords = swizzle * vec3(gl_FragCoord.xy, gl_FragCoord.z * viewportSize.x);
    //imageStore(TexVoxel, ivec3(coords), fragColor);
	imageStore(TexVoxel, ivec3(0, 0, 0), vec4(0.5,1,0.5,1));
    
}