#pragma once

#include "Actor.h"
#include "Model.h"
#include "Material.h"
#include "Textur2d.h"
#include "ShaderProgram.h"
#include "Camera.h"
#include "FrameBufferDisplay.h"

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

class World
{
public:
    World(const World& other) = delete;
    World& operator=(const World& other) = delete;

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

    void SetWindow(GLFWwindow* window, uint32_t width, uint32_t height);
	
    static World& GetInst()
    {
        static World s_World;
        return s_World;
    }

    void Start();
    void Update();

    std::vector<Material*> LoadDefaultMaterialsForModel(Model * model);

private:

    World() { }
    ~World();

	static void MouseCallback(GLFWwindow* window, double xpos, double ypos);

    GLFWwindow* m_Window;

	/*----------------  Resources --------------*/
	ModelList m_Models;
    ActorList m_Actors;
    MaterialList m_Materials;
	ShaderList m_Shaders;
    Texture2dDict m_Textures2d;
	Texture3dList m_Textures3d;

	/*----------------  Convenience Lists --------------*/
    ComponentList m_Components;
	RendererList m_Renderers;
	FrameBufferDisplayList m_FrameBufferDisplays;
	Camera* m_MainCamera;
	Camera* m_VoxelizeCamera;

	std::chrono::time_point<std::chrono::system_clock> m_LastTime;
	std::chrono::time_point<std::chrono::system_clock> m_CurrTime;

	uint32_t m_ScreenWidth;
	uint32_t m_ScreenHeight;
	float m_MouseSensitivity;
};

