#version 330 core
in vec3 worldNormal;
in vec4 vertexColor; 
  
out vec4 fragColor;

void main()
{
    fragColor = vec4(worldNormal, 1);
} 