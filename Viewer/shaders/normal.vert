#version 330

layout (location = 0) in vec3 vert;
layout (location = 1) in vec3 normal;
layout (location = 2) in vec2 aOffset;

uniform mat4 projection;
uniform mat4 view;
uniform mat4 model;
out vec3 Normal;

void main(void)  {
  Normal = mat3(model) * normal;
  vec3 Position = vec3(model * vec4(vert.xy+aOffset,vert.z, 1.0));
  gl_Position = projection * view * vec4(Position, 1.0);
}
