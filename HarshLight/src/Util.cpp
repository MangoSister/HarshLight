#include <cstdio>
#include "Util.h"

void APIENTRY glDebugOutput(GLenum source,
	GLenum type,
	GLuint id,
	GLenum severity,
	GLsizei length,
	const GLchar* message,
	const GLvoid* userParam)
{
	// ignore non-significant error/warning codes
	if (id == 131076 || id == 131169 || id == 131185 || id == 131218 || id == 131204) return;

	fprintf(stderr, "---------------\n");
	fprintf(stderr, "Debug message (%d): %s\n", id, message);

	switch (source)
	{
	case GL_DEBUG_SOURCE_API:             fprintf(stderr, "Source: API"); break;
	case GL_DEBUG_SOURCE_WINDOW_SYSTEM:   fprintf(stderr, "Source: Window System"); break;
	case GL_DEBUG_SOURCE_SHADER_COMPILER: fprintf(stderr, "Source: Shader Compiler"); break;
	case GL_DEBUG_SOURCE_THIRD_PARTY:     fprintf(stderr, "Source: Third Party"); break;
	case GL_DEBUG_SOURCE_APPLICATION:     fprintf(stderr, "Source: Application"); break;
	case GL_DEBUG_SOURCE_OTHER:           fprintf(stderr, "Source: Other"); break;
	}
	fprintf(stderr, "\n");

	switch (type)
	{
	case GL_DEBUG_TYPE_ERROR:               fprintf(stderr, "Type: Error"); break;
	case GL_DEBUG_TYPE_DEPRECATED_BEHAVIOR: fprintf(stderr, "Type: Deprecated Behaviour"); break;
	case GL_DEBUG_TYPE_UNDEFINED_BEHAVIOR:  fprintf(stderr, "Type: Undefined Behaviour"); break;
	case GL_DEBUG_TYPE_PORTABILITY:         fprintf(stderr, "Type: Portability"); break;
	case GL_DEBUG_TYPE_PERFORMANCE:         fprintf(stderr, "Type: Performance"); break;
	case GL_DEBUG_TYPE_MARKER:              fprintf(stderr, "Type: Marker"); break;
	case GL_DEBUG_TYPE_PUSH_GROUP:          fprintf(stderr, "Type: Push Group"); break;
	case GL_DEBUG_TYPE_POP_GROUP:           fprintf(stderr, "Type: Pop Group"); break;
	case GL_DEBUG_TYPE_OTHER:               fprintf(stderr, "Type: Other"); break;
	}
	fprintf(stderr, "\n");

	switch (severity)
	{
	case GL_DEBUG_SEVERITY_HIGH:         fprintf(stderr, "Severity: high"); break;
	case GL_DEBUG_SEVERITY_MEDIUM:       fprintf(stderr, "Severity: medium"); break;
	case GL_DEBUG_SEVERITY_LOW:          fprintf(stderr, "Severity: low"); break;
	case GL_DEBUG_SEVERITY_NOTIFICATION: fprintf(stderr, "Severity: notification"); break;
	}
	fprintf(stderr, "\n\n");
}


void DrawTestTriangle()
{
    glBegin(GL_TRIANGLES);
    glColor3f(1, 0, 0);
    glVertex2f(-1.0f, -1.0f);
    glVertex2f(1.5f, -0.5f);
    glVertex2f(0.0f, 1.0f);
    glEnd();
}

void DrawPixelGrid(uint32_t mWidth, uint32_t mHeight)
{
    static uint32_t mZoom = 16;
    glColor3f(0.0f, 1.0f, 0.0f);
    glMatrixMode(GL_PROJECTION);
    glPushMatrix();
    glLoadIdentity();

    glOrtho(0.0, mWidth, 0.0, mHeight, -1.0, 1.0);


    glMatrixMode(GL_MODELVIEW);
    glLoadIdentity();

    glBegin(GL_LINES);
    for (uint32_t x = 0; x <= mWidth; x += mZoom) {
        glVertex2i(x, 0);
        glVertex2i(x, mHeight);
    }
    for (uint32_t y = 0; y <= mHeight; y += mZoom) {
        glVertex2i(0, y);
        glVertex2i(mWidth, y);
    }
    glEnd();

    // draw sample points
    glBegin(GL_POINTS);
    for (uint32_t x = 0; x <= mWidth; x += mZoom) {
        for (uint32_t y = 0; y <= mHeight; y += mZoom) {
            glVertex2f(x + mZoom*0.5f, y + mZoom*0.5f);
        }
    }
    glEnd();

    glMatrixMode(GL_PROJECTION);
    glPopMatrix();
}

glm::mat4x4 LookAtDir(const glm::vec3& dir, const glm::vec3& up)
{
    const glm::vec4 r = glm::vec4(normalize(cross(up, dir)), 0.0f);
    const glm::vec4 u = glm::vec4(normalize(cross(dir, glm::vec3(r))), 0.0f);
    const glm::vec4 f = glm::vec4(-normalize(dir), 0.0f);
    glm::mat4x4 output(1.0f);
    output[0] = r;
    output[1] = u;
    output[2] = f;

    return output;
}

glm::mat4x4 ViewMtxFromTransform(const glm::mat4x4& transform)
{
    glm::mat4x4 view_mtx(1.0f);
    view_mtx[0][0] = transform[0][0];  view_mtx[0][1] = transform[1][0];  view_mtx[0][2] = transform[2][0];
    view_mtx[1][0] = transform[0][1];  view_mtx[1][1] = transform[1][1];  view_mtx[1][2] = transform[2][1];
    view_mtx[2][0] = transform[0][2];  view_mtx[2][1] = transform[1][2];  view_mtx[2][2] = transform[2][2];
    view_mtx[3][0] = -glm::dot(transform[0], transform[3]);
    view_mtx[3][1] = -glm::dot(transform[1], transform[3]);
    view_mtx[3][2] = -glm::dot(transform[2], transform[3]);

    return view_mtx;
}
