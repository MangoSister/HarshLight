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
#ifdef _DEBUG
    assert(comp != nullptr);
#endif
    m_Components.push_back(comp);
}

const ComponentList & Actor::GetAllComponents() const
{
	return m_Components;
}

void Actor::AddRenderer(ModelRenderer* renderer)
{
#ifdef _DEBUG
    assert(renderer != nullptr);
#endif
    m_Renderers.push_back(renderer);
}

const RendererList& Actor::GetAllRenderers() const
{
    return m_Renderers;
}