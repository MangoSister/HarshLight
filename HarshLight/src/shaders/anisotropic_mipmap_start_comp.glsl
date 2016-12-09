#version 450 core

#define LOCAL_SIZE 8
layout(local_size_x = LOCAL_SIZE, local_size_y = LOCAL_SIZE, local_size_z = LOCAL_SIZE) in;

//layout (binding = 0, r32ui) readonly uniform uimage3D ImgLeaf;
layout (binding = 0) uniform sampler3D ImgLeaf;
layout (binding = 1, rgba8) writeonly uniform image3D ImgInterior[6];

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

int GetLeafOffset(ivec3 idx)
{
	return (idx.x << 2) + (idx.y << 1) + idx.z;
}

void main()
{
	//gl_GlobalInvocationID is interior texture idx

	//out-of-bound check
	if( any( greaterThanEqual( gl_GlobalInvocationID, imageSize(ImgInterior[0]) ) ) )
		return;

	ivec3 child_base = ivec3(gl_GlobalInvocationID) * ivec3(2);
	vec4 val_leaf[8];
	
	val_leaf[0] = (texelFetch(ImgLeaf, child_base + ivec3(0, 0, 0), 0)); val_leaf[0].w = sign(val_leaf[0].w);
	val_leaf[1] = (texelFetch(ImgLeaf, child_base + ivec3(0, 0, 1), 0)); val_leaf[1].w = sign(val_leaf[1].w);
	val_leaf[2] = (texelFetch(ImgLeaf, child_base + ivec3(0, 1, 0), 0)); val_leaf[2].w = sign(val_leaf[2].w);
	val_leaf[3] = (texelFetch(ImgLeaf, child_base + ivec3(0, 1, 1), 0)); val_leaf[3].w = sign(val_leaf[3].w);

	val_leaf[4] = (texelFetch(ImgLeaf, child_base + ivec3(1, 0, 0), 0)); val_leaf[4].w = sign(val_leaf[4].w);
	val_leaf[5] = (texelFetch(ImgLeaf, child_base + ivec3(1, 0, 1), 0)); val_leaf[5].w = sign(val_leaf[5].w);
	val_leaf[6] = (texelFetch(ImgLeaf, child_base + ivec3(1, 1, 0), 0)); val_leaf[6].w = sign(val_leaf[6].w);
	val_leaf[7] = (texelFetch(ImgLeaf, child_base + ivec3(1, 1, 1), 0)); val_leaf[7].w = sign(val_leaf[7].w);

	//threshold counter to binary initial opacity
	//for(uint i = 0; i < 8; i++)
	//	val_leaf[i].w = sign(val_leaf[i].w);

	//x axis
	ivec3 enters[4];
	enters[0] = ivec3(0, 0, 0);
	enters[1] = ivec3(0, 1, 0);
	enters[2] = ivec3(0, 0, 1);
	enters[3] = ivec3(0, 1, 1);
	ivec3 exits[4];
	vec4 blend_pos = vec4(0.0);
	vec4 blend_neg = vec4(0.0);
	for(uint i = 0; i < 4; i++)
	{
		exits[i] = enters[i] + ivec3(1, 0, 0);
		//alpha blending positive: enters over exits
		vec4 val_enter = val_leaf[GetLeafOffset(enters[i])];
		//alpha blending negative: exits over enters
		vec4 val_exit = val_leaf[GetLeafOffset(exits[i])];
		blend_pos += val_enter + (1.0 - val_enter.w) * val_exit;
		blend_neg += val_exit + (1.0 - val_exit.w) * val_enter;
	}
	blend_pos *= 0.25;
	blend_neg *= 0.25;
	//uint u32_blend_pos = ColorVec4ToUint(blend_pos);
	imageStore(ImgInterior[0], ivec3(gl_GlobalInvocationID), blend_pos);
	//uint u32_blend_neg = ColorVec4ToUint(blend_neg);
	imageStore(ImgInterior[1], ivec3(gl_GlobalInvocationID), blend_neg);

	//y axis
	enters[0] = ivec3(0, 0, 0);
	enters[1] = ivec3(0, 0, 1);
	enters[2] = ivec3(1, 0, 0);
	enters[3] = ivec3(1, 0, 1);
	blend_pos = vec4(0.0);
	blend_neg = vec4(0.0);
	for(uint i = 0; i < 4; i++)
	{
		exits[i] = enters[i] + ivec3(0, 1, 0);
		//alpha blending positive: enters over exits
		vec4 val_enter = val_leaf[GetLeafOffset(enters[i])];
		//alpha blending negative: exits over enters
		vec4 val_exit = val_leaf[GetLeafOffset(exits[i])];
		blend_pos += val_enter + (1.0 - val_enter.w) * val_exit;
		blend_neg += val_exit + (1.0 - val_exit.w) * val_enter;
	}
	blend_pos *= 0.25;
	blend_neg *= 0.25;
	//u32_blend_pos = ColorVec4ToUint(blend_pos);
	imageStore(ImgInterior[2], ivec3(gl_GlobalInvocationID), blend_pos);
	//u32_blend_neg = ColorVec4ToUint(blend_neg);
	imageStore(ImgInterior[3], ivec3(gl_GlobalInvocationID), blend_neg);

	//z axis
	enters[0] = ivec3(0, 0, 0);
	enters[1] = ivec3(0, 0, 1);
	enters[2] = ivec3(0, 1, 0);
	enters[3] = ivec3(0, 1, 1);
	blend_pos = vec4(0.0);
	blend_neg = vec4(0.0);
	for(uint i = 0; i < 4; i++)
	{
		exits[i] = enters[i] + ivec3(0, 0, 1);
		//alpha blending positive: enters over exits
		vec4 val_enter = val_leaf[GetLeafOffset(enters[i])];
		//alpha blending negative: exits over enters
		vec4 val_exit = val_leaf[GetLeafOffset(exits[i])];
		blend_pos += val_enter + (1.0 - val_enter.w) * val_exit;
		blend_neg += val_exit + (1.0 - val_exit.w) * val_enter;
	}
	blend_pos *= 0.25;
	blend_neg *= 0.25;
	//u32_blend_pos = ColorVec4ToUint(blend_pos);
	imageStore(ImgInterior[4], ivec3(gl_GlobalInvocationID), blend_pos);
	//u32_blend_neg = ColorVec4ToUint(blend_neg);
	imageStore(ImgInterior[5], ivec3(gl_GlobalInvocationID), blend_neg);
}