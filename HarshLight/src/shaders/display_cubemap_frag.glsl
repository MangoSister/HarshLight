#version 450 core

in vec2 vs_Texcoord;
out vec4 fragColor;

uniform samplerCube TexCube;

uniform int Face;

vec3 GetCubemapCoord(in vec2 face_uv, in int face)
{
  vec3 v;
  switch(face)
  {
    case 0: v = vec3( 1.0, -face_uv.y, -face_uv.x); break; // +X
    case 1: v = vec3(-1.0, -face_uv.y,  face_uv.x); break; // -X
    case 2: v = vec3( face_uv.x,  1.0, face_uv.y); break; // +Y
    case 3: v = vec3(-face_uv.x, -1.0, face_uv.y); break; // -Y
    case 4: v = vec3( face_uv.x, -face_uv.y,  1.0); break; // +Z
    case 5: v = vec3(-face_uv.x, -face_uv.y, -1.0); break; // -Z
  }
  return v;//normalize(v);
}

void main()
{ 
	fragColor.xyz = texture(TexCube, GetCubemapCoord(vs_Texcoord * vec2(2.0) - vec2(1.0), Face)).xxx;
	fragColor.w = 1.0;
}