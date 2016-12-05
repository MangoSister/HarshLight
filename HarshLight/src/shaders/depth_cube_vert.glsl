#version 450 core
layout (location = 0) in vec3 app_Position;
layout (location = 1) in vec3 app_Normal;
layout (location = 2) in vec2 app_Uv;

uniform mat4 Model;

void main()
{
    gl_Position = Model * vec4(app_Position, 1.0);
}
