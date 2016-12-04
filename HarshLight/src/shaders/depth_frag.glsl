#version 450 core
  
out vec4 fragColor;

void main()
{
	gl_FragDepth = gl_FragCoord.z;
} 