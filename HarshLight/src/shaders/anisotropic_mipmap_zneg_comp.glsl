#version 450 core

#define LOCAL_SIZE 8
#define BLOCK_SIZE 1024

layout(local_size_x = LOCAL_SIZE, local_size_y = LOCAL_SIZE, local_size_z = LOCAL_SIZE) in;

//layout (binding = 0, rgba8) readonly volatile uniform image3D ImgChild;
layout (binding = 0) uniform sampler3D ImgChild;
layout (binding = 1, rgba8) writeonly uniform image3D ImgParent;

uniform int ChildMipLevel;
//shared vec4 s_Block[BLOCK_SIZE];

void main()
{
	//gl_GlobalInvocationID is interior texture idx

	//out-of-bound check
	if( any( greaterThanEqual( gl_GlobalInvocationID, imageSize(ImgParent) ) ) )
		return;
	
	ivec3 child_base = ivec3(gl_GlobalInvocationID) * ivec3(2);
	vec4 val_child[8];
	val_child[0] = texelFetch(ImgChild, child_base + ivec3(0, 0, 0), ChildMipLevel);
	val_child[1] = texelFetch(ImgChild, child_base + ivec3(0, 0, 1), ChildMipLevel);
	val_child[2] = texelFetch(ImgChild, child_base + ivec3(0, 1, 0), ChildMipLevel);
	val_child[3] = texelFetch(ImgChild, child_base + ivec3(0, 1, 1), ChildMipLevel);

	val_child[4] = texelFetch(ImgChild, child_base + ivec3(1, 0, 0), ChildMipLevel);
	val_child[5] = texelFetch(ImgChild, child_base + ivec3(1, 0, 1), ChildMipLevel);
	val_child[6] = texelFetch(ImgChild, child_base + ivec3(1, 1, 0), ChildMipLevel);
	val_child[7] = texelFetch(ImgChild, child_base + ivec3(1, 1, 1), ChildMipLevel);

	vec4 val_parent = vec4(0.0);
	val_parent += val_child[1] + (1.0 - val_child[1].w) * val_child[0];
	val_parent += val_child[3] + (1.0 - val_child[3].w) * val_child[2];
	val_parent += val_child[5] + (1.0 - val_child[5].w) * val_child[4];
	val_parent += val_child[7] + (1.0 - val_child[7].w) * val_child[6];
	val_parent *= 0.25;
	imageStore(ImgParent, ivec3(gl_GlobalInvocationID), val_parent);
}