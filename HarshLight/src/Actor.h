#pragma once

#include <vector>
#include "Component.h"
#include "ModelRenderer.h"

typedef std::vector<Component*> ComponentList;
typedef std::vector<ModelRenderer*> RendererList;

class Actor
{
public:

    explicit Actor();
    ~Actor();
    Actor(const Actor& other) = delete;
    Actor& operator=(const Actor& other) = delete;

    void AddComponent(Component* comp);
	const ComponentList& GetAllComponents() const;

    void AddRenderer(ModelRenderer* renderer);
    const RendererList& GetAllRenderers() const;

    template<typename Comp>
    Comp* GetComponent()
    {
        for (size_t i = 0; i < m_Components.size(); i++) 
        {
            if (Comp* comp = dynamic_cast<Comp*>(m_Components[i]))
                return comp;
        }
        return nullptr;
    }

    template<typename Rend>
    Rend* GetRenderer()
    {
        for (size_t i = 0; i < m_Renderers.size(); i++)
        {
            if (Rend* rend = dynamic_cast<Rend*>(m_Renderers[i]))
                return rend;
        }
        return nullptr;
    }

private:

    ComponentList m_Components;
    RendererList m_Renderers;
};