#include "World.h"
#include <cassert>

World::~World()
{
    for (Actor*& actor : m_Actors)
    {
        if (actor)
        {
            delete actor;
            actor = nullptr;
        }
    }

	for (Model*& model : m_Models)
	{
		if (model)
		{
			delete model;
			model = nullptr;
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

const ModelList& World::GetModels() const
{
	return m_Models;
}

void World::AddModel(Model* model)
{
	assert(model != nullptr);
	m_Models.push_back(model);
}

void World::Start()
{
    for (Actor* actor : m_Actors)
    {
        assert(actor != nullptr);
        actor->Start();
    }

	m_LastTime = std::chrono::system_clock::now();
	m_CurrTime = std::chrono::system_clock::now();
}

void World::Update()
{
	m_CurrTime = std::chrono::system_clock::now();
	float elapsed = static_cast<float>((std::chrono::duration<double>(m_CurrTime - m_LastTime)).count());
	//fps: 1 / elapsed
	m_LastTime = m_CurrTime;

    for (Actor* actor : m_Actors)
    {
        assert(actor != nullptr);
        actor->Update(elapsed);
    }
}

int World::GetKey(int key)
{
    return glfwGetKey(m_Window, key);
}