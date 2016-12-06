#version 450 core

in vec4 gs_WorldPosition;
in flat int gs_Face;

layout (std140, binding = 3) uniform LightCapture
{
	mat4 CubeLightMtx[6];
	mat4 CubeLightProjMtx;
	vec4 PointLightWorldPos;
	vec2 CaptureRange;
};

out vec4 fragColor; //dummy out

void main()
{
 //    //get distance between fragment and light source
 //   float light_dist = length(gs_WorldPosition.xyz - PointLightWorldPos.xyz);
 //   light_dist = clamp(light_dist, CaptureRange.x, CaptureRange.y);
 //    //map to [0;1] range by dividing by far_plane
 //   light_dist = (light_dist - CaptureRange.x) / (CaptureRange.y - CaptureRange.x);
	//light_dist = clamp(light_dist, 0.0, 1.0);
	//gl_FragDepth = light_dist;//Write this as modified depth

	//float depth = (CaptureRange.x + CaptureRange.y) - (2 * CaptureRange.x * CaptureRange.y) / light_dist;
	//depth /= (CaptureRange.y - CaptureRange.x);

	//depth = clamp(depth, 0.0, 1.0);
	//vec4 proj_pos = CubeLightMtx[gs_Face] * gs_WorldPosition;
	//proj_pos = inverse(CubeLightMtx[gs_Face]) * proj_pos;

	//float light_dist = length(proj_pos.xyz - PointLightWorldPos.xyz);
 //   light_dist = clamp(light_dist, CaptureRange.x, CaptureRange.y);
 //    //map to [0;1] range by dividing by far_plane
 //   light_dist = (light_dist - CaptureRange.x) / (CaptureRange.y - CaptureRange.x);
	//light_dist = clamp(light_dist, 0.0, 1.0);
	//gl_FragDepth = light_dist;//Write this as modified depth

	//vec4 proj_pos = CubeLightMtx[gs_Face] * vec4(0,0,0,1);
 //   gl_FragDepth = (proj_pos.z / proj_pos.w) * 0.5 + 0.5;

	vec4 view_pos = CubeLightMtx[gs_Face] * gs_WorldPosition;
	//vec4 proj_pos = CubeLightProjMtx * view_pos;
	//vec4 ndc_pos = proj_pos / proj_pos.w;

	gl_FragDepth = clamp((-view_pos.z - CaptureRange.x) / (CaptureRange.y - CaptureRange.x), 0.0, 1.0);

	fragColor = vec4(0, 0, 0, 1);
} 