#include "ModelRenderer.h"

ModelRenderer::ModelRenderer(Model * model, Material * material):
    Component(), m_Model(model), m_Material(material)
{
}

void ModelRenderer::Start()
{

}

void ModelRenderer::Update(float dt)
{
    m_Model->Render(m_Material);
}
