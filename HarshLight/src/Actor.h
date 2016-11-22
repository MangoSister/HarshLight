#pragma once

#include <vector>
#include "Component.h"
#include "ModelRenderer.h"

typedef std::vector<Component*> ComponentList;

class Actor
{
public:

    explicit Actor();
    ~Actor();
    Actor(const Actor& other) = delete;
    Actor& operator=(const Actor& other) = delete;

    void Start();
    void Update(float dt);

    void AddComponent(Component* comp);
	const ComponentList& GetAllComponents() const;

	ModelRenderer* GetRenderer();

private:
    ComponentList m_Components;
	ModelRenderer m_Renderer;
};