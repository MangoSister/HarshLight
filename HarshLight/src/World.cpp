#include "World.h"
#include <cassert>

World::~World()
{
    for (Actor* actor : m_Actors)
    {
        if (actor)
        {
            delete actor;
            actor = nullptr;
        }
    }
}

void World::SetWindow(GLFWwindow* window)
{
    m_Window = window;
}

const ActorList& World::GetActors() const
{
    return m_Actors;
}

void World::AddActor(Actor* actor)
{
    assert(actor != nullptr);
    m_Actors.push_back(actor);
}

void World::Start()
{
    for (Actor* actor : m_Actors)
    {
        assert(actor != nullptr);
        actor->Start();
    }
}

void World::Update(float dt)
{
    for (Actor* actor : m_Actors)
    {
        assert(actor != nullptr);
        actor->Update(dt);
    }
}

int World::GetKey(int key)
{
    return glfwGetKey(m_Window, key);
}