#pragma once
#include "ModelRenderer.h"

class FrameBufferDisplay : public ModelRenderer
{
public:
	explicit FrameBufferDisplay(Model* model, uint32_t res, uint8_t depth_test, uint8_t culling, GLuint poly_mode);
	FrameBufferDisplay(const FrameBufferDisplay& other) = delete;

	virtual ~FrameBufferDisplay();

	void StartRenderToFrameBuffer();

	GLuint GetColorBuffer() const;
private:

	uint32_t m_Dim;
	GLuint m_FBO;
	GLuint m_ColorBuffer;
	GLuint m_RBO;

    uint8_t m_EnableDepthTest;
    uint8_t m_EnableCulling;
    GLuint m_PolygonMode;
};