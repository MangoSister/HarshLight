#pragma once
#include "ModelRenderer.h"

class FrameBufferDisplay : public ModelRenderer
{
public:
	explicit FrameBufferDisplay(Model* model, uint32_t res);
	FrameBufferDisplay(const FrameBufferDisplay& other) = delete;

	virtual ~FrameBufferDisplay();

	void StartRenderToFrameBuffer();

	GLuint GetColorBuffer() const;
private:

	uint32_t m_Dim;
	GLuint m_FBO;
	GLuint m_ColorBuffer;
	GLuint m_RBO;
};