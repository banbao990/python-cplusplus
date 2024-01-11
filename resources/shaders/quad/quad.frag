#version 330 core

out vec4 FragColor;

in vec2 texCoords;
uniform sampler2D Image;

void main() {
    FragColor = texture(Image, texCoords);
}