#version 330 core
layout (location = 0) in vec3 position;
layout (location = 1) in vec3 normal;
layout (location = 2) in vec2 uv;

uniform sampler2D TexAlbedo;

out vec2 texcoord;
out vec3 worldNormal;

layout (std140) uniform CamMtx
{
    mat4 view;
    mat4 proj;
};

uniform mat4 model;

void main()
{
    gl_Position =  proj * view * model * vec4(position, 1.0);
	texcoord = uv;
	worldNormal = normalize(mat3(model) * normal);
}