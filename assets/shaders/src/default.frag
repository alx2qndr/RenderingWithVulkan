#version 450

layout(binding = 1) uniform sampler2D textureSampler;

layout(location = 0) in vec3 fragColor;
layout(location = 1) in vec2 fragTextureCoordinate;
layout(location = 2) in vec3 fragNormal;
layout(location = 3) in vec3 fragWorldPosition;

layout(location = 0) out vec4 outColor;

const vec3 albedo = vec3(0.8, 0.8, 0.8);
const vec3 lightDirection = normalize(vec3(1.0, 1.0, 1.0));
const vec3 lightColor = vec3(1.0, 1.0, 1.0);
const vec3 ambientColor = vec3(0.1, 0.1, 0.1);
const vec3 cameraViewPosition = vec3(0.0, 0.0, 2.0);
const float metallic = 0.0;
const float roughness = 0.25;

void main() {
    vec3 normal = normalize(fragNormal);
    vec3 viewDirection = normalize(cameraViewPosition - fragWorldPosition);
    
    vec3 diffuse = max(dot(normal, lightDirection), 0.0) * albedo; 
    
    vec3 halfwayDirection = normalize(lightDirection + viewDirection);
    float spec = pow(max(dot(normal, halfwayDirection), 0.0), (1.0 - roughness) * 128.0);
    vec3 specular = mix(diffuse, lightColor, metallic) * spec;

    vec3 reflectedDir = reflect(-viewDirection, normal);
    float reflectionFactor = max(dot(reflectedDir, lightDirection), 0.0);
    vec3 reflection = reflectionFactor * lightColor * metallic;

    vec3 lighting = ambientColor + diffuse + specular + reflection;
    vec3 textureColor = texture(textureSampler, fragTextureCoordinate).rgb;

    outColor = vec4(albedo * lighting, 1.0);
}