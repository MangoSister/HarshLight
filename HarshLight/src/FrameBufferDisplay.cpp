#include "FrameBufferDisplay.h"

FrameBufferDisplay::FrameBufferDisplay(Model * model, uint32_t res, GLuint poly_mode)
	:ModelRenderer(model), m_FBO(0), m_RBO(0), m_ColorBuffer(0), m_Dim(res), m_PolygonMode(poly_mode)
{
	//now model should only be quad!!!

	glGenFramebuffers(1, &m_FBO);
	glBindFramebuffer(GL_FRAMEBUFFER, m_FBO);

	glGenTextures(1, &m_ColorBuffer);
	glBindTexture(GL_TEXTURE_2D, m_ColorBuffer);
	glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, res, res, 0, GL_RGB, GL_UNSIGNED_BYTE, nullptr);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
	glBindTexture(GL_TEXTURE_2D, 0);

	glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, m_ColorBuffer, 0);
	// Create a renderbuffer object for depth and stencil attachment (we won't be sampling these)
	glGenRenderbuffers(1, &m_RBO);
	glBindRenderbuffer(GL_RENDERBUFFER, m_RBO);
	glRenderbufferStorage(GL_RENDERBUFFER, GL_DEPTH24_STENCIL8, res, res); // Use a single renderbuffer object for both a depth AND stencil buffer.
	glBindRenderbuffer(GL_RENDERBUFFER, 0);
	glFramebufferRenderbuffer(GL_FRAMEBUFFER, GL_DEPTH_STENCIL_ATTACHMENT, GL_RENDERBUFFER, m_RBO); // Now actually attach it
																								  // Now that we actually created the framebuffer and added all attachments we want to check if it is actually complete now
	if (glCheckFramebufferStatus(GL_FRAMEBUFFER) != GL_FRAMEBUFFER_COMPLETE)
	{
		fprintf(stderr, "ERROR: Framebuffer is not complete!\n");
	}
	glBindFramebuffer(GL_FRAMEBUFFER, 0);

}

FrameBufferDisplay::~FrameBufferDisplay()
{
	if (m_FBO)
	{
		glDeleteFramebuffers(1, &m_FBO);
		m_FBO = 0;
	}

	if (m_RBO)
	{
		glDeleteRenderbuffers(1, &m_RBO);
		m_RBO = 0;
	}

	if (m_ColorBuffer)
	{
		glDeleteTextures(1, &m_ColorBuffer);
		m_ColorBuffer = 0;
	}
}

void FrameBufferDisplay::StartRenderToFrameBuffer()
{
	glBindFramebuffer(GL_FRAMEBUFFER, m_FBO);
	// Clear all attached buffers        
	glClearColor(0.5f, 0.3f, 0.2f, 1.0f);
	glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT); //not using stencil buffer?

    glDisable(GL_DEPTH_TEST);
    glDisable(GL_CULL_FACE);
    glPolygonMode(GL_FRONT_AND_BACK, m_PolygonMode);
	glViewport(0, 0, m_Dim, m_Dim);
}

GLuint FrameBufferDisplay::GetColorBuffer() const
{
	return m_ColorBuffer;
}


