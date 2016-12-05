#version 450 core
in vec2 vs_TexCoords;
out vec4 fragColor;

uniform sampler2D TexFont;
uniform vec3 TextColor;

void main()
{    
    vec4 sampled = vec4(1.0, 1.0, 1.0, texture(TexFont, vs_TexCoords).r);
    fragColor = vec4(TextColor, 1.0) * sampled;
}  