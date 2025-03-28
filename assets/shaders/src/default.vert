#version 450

layout(binding = 0) uniform UniformBufferObject {
    mat4 model;
    mat4 view;
    mat4 projection;
} ubo;

layout(location = 0) in vec3 inPosition;
layout(location = 1) in vec3 inColor;
layout(location = 2) in vec2 inTextureCoordinate;
layout(location = 3) in vec3 inNormal;

layout(location = 0) out vec3 fragColor;
layout(location = 1) out vec2 fragTextureCoordinate;
layout(location = 2) out vec3 fragNormal;
layout(location = 3) out vec3 fragWorldPosition;

void main() {
    vec4 worldPosition = ubo.model * vec4(inPosition, 1.0);
    gl_Position = ubo.projection * ubo.view * worldPosition;

    fragColor = inColor;
    fragTextureCoordinate = inTextureCoordinate;

    fragNormal = mat3(transpose(inverse(ubo.model))) * inNormal;

    
    fragWorldPosition = worldPosition.xyz;
}
