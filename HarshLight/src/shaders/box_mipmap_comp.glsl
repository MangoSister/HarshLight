#version 450 core

#define LOCAL_SIZE 8
#define BLOCK_SIZE 1024

layout(local_size_x = LOCAL_SIZE, local_size_y = LOCAL_SIZE, local_size_z = LOCAL_SIZE) in;

layout (binding = 0, rgba8) readonly volatile uniform image3D ImgChild;
layout (binding = 1, rgba8) writeonly volatile uniform image3D ImgParent;

//shared vec4 s_Block[BLOCK_SIZE];

void main()
{
	//gl_GlobalInvocationID is interior texture idx

	//out-of-bound check
	if( any( greaterThanEqual( gl_GlobalInvocationID, imageSize(ImgParent) ) ) )
		return;
	
	ivec3 child_base = ivec3(gl_GlobalInvocationID) * ivec3(2);
	vec4 val_parent = vec4(0.0);
	val_parent += 0.125 * imageLoad(ImgChild, child_base + ivec3(0, 0, 0));
	val_parent += 0.125 * imageLoad(ImgChild, child_base + ivec3(0, 0, 1));
	val_parent += 0.125 * imageLoad(ImgChild, child_base + ivec3(0, 1, 0));
	val_parent += 0.125 * imageLoad(ImgChild, child_base + ivec3(0, 1, 1));

	val_parent += 0.125 * imageLoad(ImgChild, child_base + ivec3(1, 0, 0));
	val_parent += 0.125 * imageLoad(ImgChild, child_base + ivec3(1, 0, 1));
	val_parent += 0.125 * imageLoad(ImgChild, child_base + ivec3(1, 1, 0));
	val_parent += 0.125 * imageLoad(ImgChild, child_base + ivec3(1, 1, 1));

	imageStore(ImgParent, ivec3(gl_GlobalInvocationID), val_parent);
}