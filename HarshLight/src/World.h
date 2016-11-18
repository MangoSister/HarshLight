#pragma once

#include "Actor.h"
#include "Model.h"
#include "Material.h"
#include "GLFW/glfw3.h"
#include <chrono>
#include <vector>

typedef std::vector<Actor*> ActorList;
typedef std::vector<Model*> ModelList;
typedef std::vector<Material*> MaterialList;

class World
{
public:
    World(const World& other) = delete;
    World& operator=(const World& other) = delete;

    int GetKey(int key);
    const ActorList& GetActors() const;
    void AddActor(Actor* actor);
	const ModelList& GetModels() const;
	void AddModel(Model* model);
    const MaterialList& GetMaterials() const;
    void AddMaterial(Material* material);


    void SetWindow(GLFWwindow* window);
	
    static World& GetInst()
    {
        static World s_World;
        return s_World;
    }

    void Start();
    void Update();

private:

    World() { }
    ~World();

    GLFWwindow* m_Window;
	ModelList m_Models;
    ActorList m_Actors;
    MaterialList m_Materials;

	std::chrono::time_point<std::chrono::system_clock> m_LastTime;
	std::chrono::time_point<std::chrono::system_clock> m_CurrTime;
};

