#pragma once

#include "Actor.h"
#include "Model.h"
#include "Material.h"
#include "Textur2d.h"
#include "Camera.h"

#include "GLFW/glfw3.h"
#include <chrono>
#include <vector>
#include <unordered_map>
#include <string>

typedef std::vector<Actor*> ActorList;
typedef std::vector<Model*> ModelList;
typedef std::vector<Material*> MaterialList;
typedef std::vector<Texture2d*> Texture2dList;
typedef std::unordered_map<std::string, Texture2d*> Texture2dDict;

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
    const MaterialList& GetMaterials() const;
    void RegisterMaterial(Material* material);
    const Texture2dDict& GetTexture2ds() const;
    void RegisterTexture2d(const std::string& path, Texture2d* tex2d);

    Camera* GetMainCamera();
	void SetMainCamera(Camera* camera);


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
	ModelList m_Models;
    ActorList m_Actors;
    MaterialList m_Materials;
    Texture2dDict m_Texture2ds;

	Camera* m_MainCamera;

	std::chrono::time_point<std::chrono::system_clock> m_LastTime;
	std::chrono::time_point<std::chrono::system_clock> m_CurrTime;

	uint32_t m_ScreenWidth;
	uint32_t m_ScreenHeight;
};

