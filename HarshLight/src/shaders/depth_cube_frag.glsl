#version 450 core

in vec4 gs_WorldPosition;

layout (std140, binding = 3) uniform LightCapture
{
	mat4 CubeLightMtx[6];
	uniform vec4 PointLightWorldPos;
	uniform vec2 CaptureRange;
};

void main()
{
    // get distance between fragment and light source
    float light_dist = length(gs_WorldPosition.xyz - PointLightWorldPos.xyz);
    
    // map to [0;1] range by dividing by far_plane
    light_dist = (light_dist - CaptureRange.x) / (CaptureRange.y - CaptureRange.x);
	light_dist = clamp(light_dist, 0.0, 1.0);

    // Write this as modified depth
    gl_FragDepth = light_dist;
} 