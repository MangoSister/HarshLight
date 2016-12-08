#version 450 core
layout (location = 0) in vec3 app_Position;
layout (location = 1) in vec3 app_Normal;
layout (location = 2) in vec2 app_Uv;
layout (location = 3) in vec3 app_Tangent;


out vec2 vs_Texcoord;
out vec3 vs_WorldPosition;
out vec3 vs_WorldNormal;
out vec3 vs_WorldTangent;

layout (std140, binding = 0) uniform MainCamMtx
{
    mat4 View;
    mat4 Proj;
	vec4 CamWorldPos;
};

uniform mat4 Model;

void main()
{
    gl_Position = Proj * View * Model * vec4(app_Position, 1.0);
	vs_Texcoord = app_Uv;
	vs_WorldPosition = vec3(Model * vec4(app_Position, 1.0));
	vs_WorldNormal = (transpose(inverse(Model)) * vec4(app_Normal, 0.0)).xyz;
	vs_WorldNormal = normalize(vs_WorldNormal);
	vs_WorldTangent = (Model * vec4(app_Tangent, 1.0)).xyz;
}