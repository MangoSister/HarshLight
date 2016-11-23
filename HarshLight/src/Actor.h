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

private:

    ComponentList m_Components;
    RendererList m_Renderers;
};