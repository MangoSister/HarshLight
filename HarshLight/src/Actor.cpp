#include "Actor.h"
#include <cassert>

Actor::Actor(): m_Components() { }

Actor::~Actor()
{
    for (Component*& comp : m_Components)
    {
        if (comp)
        {
            delete comp;
            comp = nullptr;
        }
    }
}

void Actor::AddComponent(Component* comp)
{
    assert(comp != nullptr);
    m_Components.push_back(comp);
}

void Actor::Start()
{
    for (Component* comp : m_Components)
    {
        assert(comp != nullptr);
        comp->Start();
        comp->MarkStarted();
    }
}

void Actor::Update(float dt)
{
    for (Component* comp : m_Components)
    {
        assert(comp != nullptr);
        if (!comp->IsStarted())
        {
            comp->Start();
            comp->MarkStarted();
        }
        comp->Update(dt);
    }
}