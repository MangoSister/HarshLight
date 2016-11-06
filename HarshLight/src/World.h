#include "Actor.h"
#include "GLFW/glfw3.h"
#include <vector>
#pragma once

typedef std::vector<Actor*> ActorList;

class World
{
public:
    World(const World& other) = delete;
    World& operator=(const World& other) = delete;

    int GetKey(int key);
    const ActorList& GetActors() const;
    void AddActor(Actor* actor);
    void SetWindow(GLFWwindow* window);

    static World& GetInst()
    {
        World s_World;
        return s_World;
    }

    void Start();
    void Update(float dt);

private:

    World() { }
    ~World();

    GLFWwindow* m_Window;
    ActorList m_Actors;

};

