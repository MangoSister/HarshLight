#pragma once

#include "Util.h"
#include "Actor.h"
#include "Model.h"
#include "Material.h"
#include "Texture.h"
#include "ShaderProgram.h"
#include "Camera.h"
#include "FrameBufferDisplay.h"
#include "VoxelizeController.h"
#include "Light.h"
#include "Text.h"

#include "GLFW/glfw3.h"
#include <chrono>
#include <vector>
#include <unordered_map>
#include <string>

typedef std::vector<Actor*> ActorList;
typedef std::vector<Model*> ModelList;
typedef std::vector<Material*> MaterialList;
typedef std::unordered_map<std::string, Texture2d*> Texture2dDict;
typedef std::vector<Texture3dCompute*> Texture3dList;
typedef std::vector<ShaderProgram*> ShaderList;
typedef std::vector<ModelRenderer*> RendererList;
typedef std::vector<FrameBufferDisplay*> FrameBufferDisplayList;
typedef std::unordered_map<int, int> KeyStatusMap;

class World
{
public:
    World(const World& other) = delete;
    World& operator=(const World& other) = delete;

	void Destroy();

    bool IsKeyDown(int key);
    int GetKey(int key);
    
	const ActorList& GetActors() const;
    void RegisterActor(Actor* actor);
	
	const ModelList& GetModels() const;
	void RegisterModel(Model* model);
    
	const ShaderList& GetShaders() const;
    void RegisterShader(ShaderProgram* shader);
	
	const MaterialList& GetMaterials() const;
	void RegisterMaterial(Material* material);

    const Texture2dDict& GetTexture2ds() const;
    void RegisterTexture2d(const std::string& path, Texture2d* tex2d);
	
	const Texture3dList& GetTexture3ds() const;
	void RegisterTexture3d(Texture3dCompute* tex3d);

	Camera* GetVoxelCamera() const;
	void SetVoxelCamera(Camera* camera);

	Camera* GetMainCamera() const;
	void SetMainCamera(Camera* camera);
	void SetMouseSensitivity(float sensitivity);

	const RendererList& GetRenderers() const;

    LightManager& GetLightManager();

	const void GetFullRenderSize(uint32_t& width, uint32_t& height) const;

    void SetWindow(GLFWwindow* window, uint32_t width, uint32_t height);
	
    static World& GetInst()
    {
        static World s_World;
        return s_World;
    }

    void Start();
    void MainLoop();

    std::vector<Material*> LoadDefaultMaterialsForModel(Model * model);

	VoxelizeController* m_VoxelizeController;
	Texture2d* m_DefaultBlackTex;
	Texture2d* m_DefaultWhiteTex;
	Texture2d* m_DefaultGrayTex;
	Texture2d* m_DefaultNormalTex;
    ModelRenderer* m_DeferredShadingQuad;

private:

    World() { }
	~World() { }

	static void MouseCallback(GLFWwindow* window, double xpos, double ypos);
    static void KeyboardCallback(GLFWwindow* window, int key, int scancode, int action, int mods);

    GLFWwindow* m_Window;

    void ComputeGeometryPass();
    void ComputeShadingPass();
	void RenderUIText();

	/*----------------  Resources --------------*/
	ModelList m_Models;
    ActorList m_Actors;
    MaterialList m_Materials;
	ShaderList m_Shaders;
    Texture2dDict m_Textures2d;
	Texture3dList m_Textures3d;
    LightManager m_LightManager;
    TextManager m_TextManager;

	/*----------------  Convenience Lists --------------*/
    ComponentList m_Components;
	RendererList m_Renderers;
	FrameBufferDisplayList m_FrameBufferDisplays;
	Camera* m_MainCamera;
	Camera* m_VoxelizeCamera;

    KeyStatusMap m_KeyStatusMap;

	std::chrono::time_point<std::chrono::system_clock> m_LastTime;
	std::chrono::time_point<std::chrono::system_clock> m_CurrTime;
	float m_CurrDeltaTime;

	uint32_t m_FullRenderWidth;
	uint32_t m_FullRenderHeight;
	float m_MouseSensitivity;

    uint8_t m_RenderPassSwitch[2] = { 1, 0 };

    /*----------------  G-buffer --------------*/
    GLuint m_GBufferFBO = 0;
    GLuint m_GPositionAndSpecPower = 0; //RGBA 16F HERE 
    GLuint m_GNormalAndTangent = 0; //RGBA 16F HERE 
    GLuint m_GAlbedoAndSpecIntensity = 0; //RGBA 8 HERE
    GLuint m_GDepthRBO = 0;
	GLuint m_IndirectDiffuseHalfTex;
};