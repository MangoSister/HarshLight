#version 450 core
layout (location = 0) in vec3 position;
layout (location = 1) in vec3 normal;
layout (location = 2) in vec2 uv;

out vec3 worldNormal;
out vec4 vertexColor; 

layout (std140) uniform CamMtx
{
    mat4 view;
    mat4 proj;
};

uniform mat4 model;

void main()
{
    gl_Position =  proj * view * model * vec4(position, 1.0);
    vertexColor = vec4(0.5, 0.0, 0.0, 1.0);
	worldNormal = normalize(mat3(model) * normal);
}