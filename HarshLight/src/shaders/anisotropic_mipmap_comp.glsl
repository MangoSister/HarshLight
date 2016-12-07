#version 450 core

#define LOCAL_SIZE 8
layout(local_size_x = LOCAL_SIZE, local_size_y = LOCAL_SIZE, local_size_z = LOCAL_SIZE) in;

layout (binding = 0, r32ui) readonly volatile uniform uimage3D ImgFine[2];
layout (binding = 1, r32ui) writeonly volatile uniform uimage3D ImgCoarse[2];

uniform ivec3 ChildBaseOffset;
uniform ivec3 OrthoDirections[2];
uniform ivec3 MarchDirection;

uint ColorVec4ToUint(vec4 val) 
{
	val *= 255.0;
	return	(uint(val.x) & 0x000000FF) << 24U | 
			(uint(val.y) & 0x000000FF) << 16U | 
			(uint(val.z) & 0x000000FF) << 8U | 
			(uint(val.w) & 0x000000FF);
}


vec4 ColorUintToVec4(uint val) 
{
	float r = float((val & 0xFF000000) >> 24U);
	float g = float((val & 0x00FF0000) >> 16U);
	float b = float((val & 0x0000FF00) >> 8U);
	float a = float((val & 0x000000FF));

	vec4 o = vec4(r, g, b, a);
	o /= 255.0;
	o = clamp(o, vec4(0.0, 0.0, 0.0, 0.0), vec4(1.0, 1.0, 1.0, 1.0));
	return o;
}

void main()
{
	//gl_GlobalInvocationID is coarse texture idx

	//out-of-bound check
	if( any( greaterThanEqual( gl_GlobalInvocationID, imageSize(ImgCoarse[0]) ) ) )
		return;

	//uvec3 child_base = gl_GlobalInvocationID * uvec3(2) + ChildBaseOffset;

	ivec3 enters[4];
	enters[0] = ivec3(gl_GlobalInvocationID) * ivec3(2) + ChildBaseOffset; // child_base + child_base_offset
	enters[1] = enters[0] + OrthoDirections[0];
	enters[2] = enters[0] + OrthoDirections[1];
	enters[3] = enters[1] + OrthoDirections[1];
	ivec3 exits[4];
	vec4 blend = vec4(0.0);
	for(uint i = 0; i < 4; i++)
	{
		exits[i] = enters[i] + MarchDirection;
		//alpha blending: enters over exits
		vec4 val_enter = ColorUintToVec4(imageLoad(ImgFine[0], enters[i]).x);
		vec4 val_exit = ColorUintToVec4(imageLoad(ImgFine[0], exits[i]).x);
		blend += val_enter + (1.0 - val_enter.w) * val_exit;
	}
	blend *= 0.25;	
	uint u32_blend = ColorVec4ToUint(blend);
	imageStore(ImgCoarse[0], ivec3(gl_GlobalInvocationID), uvec4(u32_blend, 0, 0, 0));

	blend = vec4(0.0);
	for(uint i = 0; i < 4; i++)
	{
		//reverse direction: exits over enters
		vec4 val_exit = ColorUintToVec4(imageLoad(ImgFine[1], exits[i]).x);
		vec4 val_enter = ColorUintToVec4(imageLoad(ImgFine[1], enters[i]).x);
		blend += val_exit + (1.0 - val_exit.w) * val_enter;
	}
	blend *= 0.25;
	u32_blend = ColorVec4ToUint(blend);
	imageStore(ImgCoarse[1], ivec3(gl_GlobalInvocationID), uvec4(u32_blend, 0, 0, 0));
}