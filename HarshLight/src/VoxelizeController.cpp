#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/type_ptr.hpp>

#include "VoxelizeController.h"
#include "ModelRenderer.h"
#include "Actor.h"
#include "World.h"
#include "Camera.h"


const char* VoxelizeController::s_VoxelDimName = "VoxelDim";
const char* VoxelizeController::s_ViewMtxToDownName = "ViewMtxToDown";
const char* VoxelizeController::s_ViewMtxToLeftName = "ViewMtxToLeft";
const char* VoxelizeController::s_ViewMtxToForwardName = "ViewMtxToForward";

const char* VoxelizeController::s_VoxelChannelNames[VoxelChannel::Count]
{
	"TexVoxelAlbedo",
	"TexVoxelNormal",
	"TexVoxelRadiance",
};

VoxelizeController::VoxelizeController(uint32_t voxel_dim, uint32_t light_injection_res, const glm::vec3& center, const glm::vec3& extent, Camera* voxel_cam)
    :Component(), m_VoxelDim(voxel_dim), m_Center(center), m_Extent(extent), m_VoxelCam(voxel_cam), m_DirLightInjectionRes(light_injection_res)
{
	m_VoxelizeTex[VoxelChannel::TexVoxelAlbedo] = new Texture3dCompute(voxel_dim, voxel_dim, voxel_dim, GL_R32UI, GL_RED_INTEGER, GL_UNSIGNED_INT, TextureFilter::kPoint);
	m_VoxelizeTex[VoxelChannel::TexVoxelNormal] = new Texture3dCompute(voxel_dim, voxel_dim, voxel_dim, GL_R32UI, GL_RED_INTEGER, GL_UNSIGNED_INT, TextureFilter::kPoint);
	m_VoxelizeTex[VoxelChannel::TexVoxelRadiance] = new Texture3dCompute(voxel_dim, voxel_dim, voxel_dim, GL_R32UI, GL_RED_INTEGER, GL_UNSIGNED_INT, TextureFilter::kBilinear);

	//for (uint32_t i = 0; i < VoxelChannel::Count; i++)
	//	m_VoxelizeTex[i] = new Texture3dCompute(voxel_dim, voxel_dim, voxel_dim, GL_R32UI, GL_RED_INTEGER, GL_UNSIGNED_INT, false);

	glGenBuffers(1, &m_DirLightViewUBuffer);
	glBindBuffer(GL_UNIFORM_BUFFER, m_DirLightViewUBuffer);
	glBufferData(GL_UNIFORM_BUFFER, Camera::GetUBufferSize(), nullptr, GL_STATIC_DRAW);
	glBindBuffer(GL_UNIFORM_BUFFER, 0);

    glGenBuffers(1, &m_PointLightCaptureUBuffer);
    glBindBuffer(GL_UNIFORM_BUFFER, m_PointLightCaptureUBuffer);
    glBufferData(GL_UNIFORM_BUFFER, GetPointLightCaptureUBufferSize(), nullptr, GL_STATIC_DRAW);
    glBindBuffer(GL_UNIFORM_BUFFER, 0);

	glGenFramebuffers(1, &m_DepthFBO);

	glGenTextures(LightManager::s_DirLightMaxNum, m_DirectionalDepthMap);
	for (uint32_t i = 0; i < LightManager::s_DirLightMaxNum; i++)
	{
		glBindTexture(GL_TEXTURE_2D, m_DirectionalDepthMap[i]);
		glTexImage2D(GL_TEXTURE_2D, 0, GL_DEPTH_COMPONENT,
			m_DirLightInjectionRes, m_DirLightInjectionRes, 0, GL_DEPTH_COMPONENT, GL_FLOAT, NULL);
		glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
		glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
		glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT);
		glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_REPEAT);
	}
	glBindTexture(GL_TEXTURE_2D, 0);

    glGenTextures(LightManager::s_PointLightMaxNum, m_CubeDepthMap);
	for (uint32_t i = 0; i < LightManager::s_PointLightMaxNum; i++)
	{
		glBindTexture(GL_TEXTURE_CUBE_MAP, m_CubeDepthMap[i]);
		for (GLuint i = 0; i < 6; i++)
			glTexImage2D(GL_TEXTURE_CUBE_MAP_POSITIVE_X + i, 0, GL_DEPTH_COMPONENT24,
				m_PointLightInjectionRes, m_PointLightInjectionRes, 0, GL_DEPTH_COMPONENT, GL_FLOAT, NULL);
		glTexParameteri(GL_TEXTURE_CUBE_MAP, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
		glTexParameteri(GL_TEXTURE_CUBE_MAP, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
		glTexParameteri(GL_TEXTURE_CUBE_MAP, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
		glTexParameteri(GL_TEXTURE_CUBE_MAP, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
		glTexParameteri(GL_TEXTURE_CUBE_MAP, GL_TEXTURE_WRAP_R, GL_CLAMP_TO_EDGE);
	}
	glBindTexture(GL_TEXTURE_CUBE_MAP, 0);

    m_DirLightInjectionShader = new ComputeShaderProgram();
    m_DirLightInjectionShader->AddShader("src/shaders/dirlight_injection_comp.glsl");
    m_DirLightInjectionShader->LinkProgram();

	m_PointLightInjectionShader = new ComputeShaderProgram();
	m_PointLightInjectionShader->AddShader("src/shaders/pointlight_injection_comp.glsl");
	m_PointLightInjectionShader->LinkProgram();

	m_AnisotropicMipmapShaderLeaf = new ComputeShaderProgram();
	m_AnisotropicMipmapShaderLeaf->AddShader("src/shaders/anisotropic_mipmap_start_comp.glsl");
	m_AnisotropicMipmapShaderLeaf->LinkProgram();

    char name[50];
    const char* axis_pf[6]
    {
      "xpos","xneg", "ypos","yneg","zpos", "zneg"
    };
    for (uint32_t i = 0; i < s_AnisotropicMipmapCount; i++)
    {
        memset(name, 0, 50);
        sprintf(name, "src/shaders/anisotropic_mipmap_%s_comp.glsl", axis_pf[i]);
        m_AnisotropicMipmapShaderInterior[i] = new ComputeShaderProgram();
        m_AnisotropicMipmapShaderInterior[i]->AddShader(name);
        m_AnisotropicMipmapShaderInterior[i]->LinkProgram();
    }

	uint32_t half_voxel_dim = m_VoxelDim / 2;
	for (uint32_t i = 0; i < s_AnisotropicMipmapCount; i++)
		m_AnisoRadianceMipmap[i] = new Texture3dCompute(half_voxel_dim, half_voxel_dim, half_voxel_dim, GL_RGBA8, GL_RGBA, GL_FLOAT, TextureFilter::kTrilinear);
}

VoxelizeController::~VoxelizeController()
{
	//unregister voxel_tex from cuda
	for (uint32_t i = 0; i < VoxelChannel::Count; i++)
	{
		cudaCheckError(cudaGraphicsUnregisterResource(m_CudaResources[i]));
		if (m_VoxelizeTex[i])
		{
			delete m_VoxelizeTex[i];
			m_VoxelizeTex[i] = nullptr;
		}
	}

	if (m_DirLightViewUBuffer)
	{
		glDeleteBuffers(1, &m_DirLightViewUBuffer);
		m_DirLightViewUBuffer = 0;
	}

    if (m_PointLightCaptureUBuffer)
    {
        glDeleteBuffers(1, &m_PointLightCaptureUBuffer);
        m_PointLightCaptureUBuffer = 0;
    }

	for (uint32_t i = 0; i < LightManager::s_DirLightMaxNum; i++)
	{
		if (m_DirectionalDepthMap[i])
		{
			glDeleteTextures(1, &m_DirectionalDepthMap[i]);
			m_DirectionalDepthMap[i] = 0;
		}
	}

	for (uint32_t i = 0; i < LightManager::s_PointLightMaxNum; i++)
	{
		if (m_CubeDepthMap[i])
		{
			glDeleteTextures(1, &m_CubeDepthMap[i]);
			m_CubeDepthMap[i] = 0;
		}
	}

	if (m_DepthFBO)
	{
		glDeleteFramebuffers(1, &m_DepthFBO);
		m_DepthFBO = 0;
	}

    if (m_DirLightInjectionShader)
    {
        delete m_DirLightInjectionShader;
        m_DirLightInjectionShader = nullptr;
    }

	if (m_PointLightInjectionShader)
	{
		delete m_PointLightInjectionShader;
		m_PointLightInjectionShader = nullptr;
	}

	for (uint32_t i = 0; i < s_AnisotropicMipmapCount; i++)
	{
		if (m_AnisoRadianceMipmap[i])
		{
			delete m_AnisoRadianceMipmap[i];
			m_AnisoRadianceMipmap[i] = nullptr;
		}
	}	
}


void VoxelizeController::Start()
{
    glm::mat4x4 old_transform = m_VoxelCam->GetTransform();

    const float max_extent = std::max(m_Extent.x, std::max(m_Extent.y, m_Extent.z));

    glm::vec3 pos = m_Center + glm::vec3(0.0f, max_extent, 0.0f);
    m_VoxelCam->MoveTo(pos);
    m_VoxelCam->LookAtDir(glm::vec3(0.0f, -1.0f, 0.0f), glm::vec3(0.0f, 0.0f, 1.0f));
    glm::mat4x4 view_to_down = m_VoxelCam->GetViewMtx();

    pos = m_Center + glm::vec3(max_extent, 0.0, 0.0f);
    m_VoxelCam->MoveTo(pos);
    m_VoxelCam->LookAtDir(glm::vec3(-1.0f, 0.0f, 0.0f), glm::vec3(0.0f, 1.0f, 0.0f));
    glm::mat4x4 view_to_left = m_VoxelCam->GetViewMtx();

    pos = m_Center + glm::vec3(0.0f, 0.0f, -max_extent);
    m_VoxelCam->MoveTo(pos);
    m_VoxelCam->LookAtDir(glm::vec3(0.0f, 0.0f, 1.0f), glm::vec3(0.0f, 1.0f, 0.0f));
    glm::mat4x4 view_to_forward = m_VoxelCam->GetViewMtx();

    m_VoxelCam->SetTransform(old_transform);

	const std::vector<Material*>& voxel_mats = m_Actor->GetRenderer<ModelRenderer>()->GetMaterial(RenderPass::kVoxelize);
	for (auto mat : voxel_mats)
	{
        if (mat)
        {
            mat->SetVec2Param(s_VoxelDimName, glm::vec2(static_cast<float>(m_VoxelDim), static_cast<float>(m_VoxelDim)));
            mat->SetMat4x4Param(s_ViewMtxToDownName, view_to_down);
            mat->SetMat4x4Param(s_ViewMtxToLeftName, view_to_left);
            mat->SetMat4x4Param(s_ViewMtxToForwardName, view_to_forward);
        }
	}

	const std::vector<Material*>& regular_mats = m_Actor->GetRenderer<ModelRenderer>()->GetMaterial(RenderPass::kRegular);
	for (auto mat : regular_mats)
	{
        if (mat)
        {
            //mat->SetVec2Param(s_VoxelDimName, glm::vec2(static_cast<float>(m_VoxelDim), static_cast<float>(m_VoxelDim)));
			mat->SetFloatParam(s_VoxelDimName, static_cast<float>(m_VoxelDim));
            mat->SetMat4x4Param(s_ViewMtxToDownName, view_to_down);
            mat->SetMat4x4Param(s_ViewMtxToLeftName, view_to_left);
            mat->SetMat4x4Param(s_ViewMtxToForwardName, view_to_forward);
        }
	}

	//register voxel texture to cuda
	for (uint32_t i = 0; i < VoxelChannel::Count; i++)
	{
		cudaCheckError(cudaGraphicsGLRegisterImage(&m_CudaResources[i], m_VoxelizeTex[i]->GetTexObj(),
			GL_TEXTURE_3D, cudaGraphicsRegisterFlagsSurfaceLoadStore));
	}
	
	DispatchVoxelization();
}

void VoxelizeController::Update(float dt)
{
    GLuint clear_color[4] = { 0,0,0,0 };
    m_VoxelizeTex[VoxelChannel::TexVoxelRadiance]->CleanContent(clear_color);
   // glMemoryBarrier(GL_ALL_BARRIER_BITS);
	DispatchDirLightInjection();
    DispatchPointLightInjection();
	MipmapRadiance();
}

void VoxelizeController::SetVoxelDim(uint32_t dim)
{
    m_VoxelDim = dim;
}

uint32_t VoxelizeController::GetVoxelDim() const
{
    return m_VoxelDim;
}

void VoxelizeController::TransferVoxelDataToCuda(cudaSurfaceObject_t surf_objs[VoxelChannel::Count])
{
	cudaCheckError(cudaGraphicsMapResources(VoxelChannel::Count, m_CudaResources, 0));
	
	for (uint32_t i = 0; i < VoxelChannel::Count; i++)
	{		
		cudaArray* channel_cuda_array;
		cudaCheckError(cudaGraphicsSubResourceGetMappedArray(&channel_cuda_array, m_CudaResources[i], 0, 0));

		struct cudaResourceDesc res_desc;
		memset(&res_desc, 0, sizeof(cudaResourceDesc));
		res_desc.resType = cudaResourceTypeArray;    // be sure to set the resource type to cudaResourceTypeArray
		res_desc.res.array.array = channel_cuda_array;

		cudaCheckError(cudaCreateSurfaceObject(&surf_objs[i], &res_desc));
	}
}

void VoxelizeController::FinishVoxelDataFromCuda(cudaSurfaceObject_t surf_objs[VoxelChannel::Count])
{
    //there is no unbinding surface API
	for (uint32_t i = 0; i < VoxelChannel::Count; i++)
		cudaCheckError(cudaDestroySurfaceObject(surf_objs[i]));

	cudaCheckError(cudaGraphicsUnmapResources(VoxelChannel::Count, m_CudaResources, 0));
}

void VoxelizeController::EnableShadowSampling()
{
	for (uint32_t i = 0; i < LightManager::s_DirLightMaxNum; i++)
	{
		glBindTexture(GL_TEXTURE_2D, m_DirectionalDepthMap[i]);
		glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
		glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
		glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_COMPARE_MODE, GL_COMPARE_REF_TO_TEXTURE);
		glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_COMPARE_FUNC, GL_LEQUAL);
	}
	glBindTexture(GL_TEXTURE_2D, 0);
}

void VoxelizeController::DisableShadowSampling()
{
	for (uint32_t i = 0; i < LightManager::s_DirLightMaxNum; i++)
	{
		glBindTexture(GL_TEXTURE_2D, m_DirectionalDepthMap[i]);
		glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
		glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
		glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_COMPARE_MODE, GL_NONE);
	}
	glBindTexture(GL_TEXTURE_2D, 0);
}

void VoxelizeController::DispatchVoxelization()
{
    GLuint clear_color[4] = { 0,0,0,0 };
	for (uint32_t i = 0; i < VoxelChannel::Count; i++)
		if (m_VoxelizeTex[i] == nullptr) return;
    //for (uint32_t i = 0; i < VoxelChannel::Count; i++)
    //    m_VoxelizeTex[i]->CleanContent(clear_color);

	glDisable(GL_DEPTH_TEST);
	glDisable(GL_CULL_FACE);
	glPolygonMode(GL_FRONT_AND_BACK, GL_FILL);
	glColorMask(GL_FALSE, GL_FALSE, GL_FALSE, GL_FALSE);
	glDepthMask(GL_FALSE);
	glViewport(0, 0, m_VoxelDim, m_VoxelDim);

	if (m_VoxelCam)
		m_VoxelCam->UpdateCamMtx(UniformBufferBinding::kMainCam);
	else
		fprintf(stderr, "WARNING: VoxelizeCamera is null\n");

	const RendererList& renderers = World::GetInst().GetRenderers();
	for (ModelRenderer* renderer : renderers)
	{
		renderer->Render(RenderPass::kVoxelize);		
	}

	glMemoryBarrier(GL_SHADER_IMAGE_ACCESS_BARRIER_BIT);

	glEnable(GL_DEPTH_TEST);
	glEnable(GL_CULL_FACE);
	glColorMask(GL_TRUE, GL_TRUE, GL_TRUE, GL_TRUE);
	glDepthMask(GL_TRUE);
	uint32_t vw, vh;
	World::GetInst().GetViewportSize(vw, vh);
	glViewport(0, 0, vw, vh);

}

void VoxelizeController::DispatchDirLightInjection()
{
	for (uint32_t i = 0; i < VoxelChannel::Count; i++)
		if (m_VoxelizeTex[i] == nullptr) return;

	LightManager& light_manager = World::GetInst().GetLightManager();
	const RendererList& renderers = World::GetInst().GetRenderers();

	DisableShadowSampling();

	glEnable(GL_DEPTH_TEST);
	glEnable(GL_CULL_FACE);
	glPolygonMode(GL_FRONT_AND_BACK, GL_FILL);
	glColorMask(GL_TRUE, GL_TRUE, GL_TRUE, GL_TRUE);
	glDepthMask(GL_TRUE);
	glViewport(0, 0, m_DirLightInjectionRes, m_DirLightInjectionRes);



	for (uint32_t i = 0; i < light_manager.GetDirLightCount(); i++)
	{
		glBindFramebuffer(GL_FRAMEBUFFER, m_DepthFBO);
		glFramebufferTexture2D(GL_FRAMEBUFFER, GL_DEPTH_ATTACHMENT, GL_TEXTURE_2D, m_DirectionalDepthMap[i], 0);
		glDrawBuffer(GL_NONE);
		glReadBuffer(GL_NONE);
#if _DEBUG
		GLenum status = glCheckFramebufferStatus(GL_FRAMEBUFFER);
		assert(status == GL_FRAMEBUFFER_COMPLETE);
#endif

		DirLight& dir_light = light_manager.GetDirLight(i);
		//overwrite main camera

		glBindBufferRange(GL_UNIFORM_BUFFER, (uint8_t)UniformBufferBinding::kMainCam, m_DirLightViewUBuffer, 0, Camera::GetUBufferSize());
		glBindBuffer(GL_UNIFORM_BUFFER, m_DirLightViewUBuffer);
		const glm::mat4x4& light_mtx = dir_light.m_LightMtx;
        //compute light space bounding box
        glm::vec3 min, max;
        LightSpaceBBox(dir_light, min, max);
        glm::mat4x4 light_proj_mtx = glm::ortho(min.x, max.x, min.y, max.y, -max.z, -min.z);

		dir_light.m_LightProjMtx = light_proj_mtx; //update dir light projection matrices here

        glBufferSubData(GL_UNIFORM_BUFFER, 0, sizeof(glm::mat4), glm::value_ptr(light_mtx));
        glBufferSubData(GL_UNIFORM_BUFFER, sizeof(glm::mat4), sizeof(glm::mat4), glm::value_ptr(light_proj_mtx));
        glBufferSubData(GL_UNIFORM_BUFFER, 2 * sizeof(glm::mat4), sizeof(glm::vec4), glm::value_ptr(glm::vec4(0.0f, 0.0f, 0.0f, 1.0f)));
		glBindBuffer(GL_UNIFORM_BUFFER, 0);

        glClear(GL_DEPTH_BUFFER_BIT);
		for (ModelRenderer* renderer : renderers)
			renderer->Render(RenderPass::kDirLightInjection);

        //now we have depth for current light, inject current light info
		glUseProgram(m_DirLightInjectionShader->GetProgram());

        GLint loc = glGetUniformLocation(m_DirLightInjectionShader->GetProgram(), s_VoxelChannelNames[VoxelChannel::TexVoxelAlbedo]);
        if(loc != -1)
            glBindImageTexture(BINDING_POINT_START_VOXEL_IMG + VoxelChannel::TexVoxelAlbedo, m_VoxelizeTex[VoxelChannel::TexVoxelAlbedo]->GetTexObj(), 0, GL_TRUE, 0, GL_READ_ONLY, m_VoxelizeTex[VoxelChannel::TexVoxelAlbedo]->GetInternalFormat());
        loc = glGetUniformLocation(m_DirLightInjectionShader->GetProgram(), s_VoxelChannelNames[VoxelChannel::TexVoxelNormal]);
        if (loc != -1)
            glBindImageTexture(BINDING_POINT_START_VOXEL_IMG + VoxelChannel::TexVoxelNormal, m_VoxelizeTex[VoxelChannel::TexVoxelNormal]->GetTexObj(), 0, GL_TRUE, 0, GL_READ_ONLY, m_VoxelizeTex[VoxelChannel::TexVoxelNormal]->GetInternalFormat());
        loc = glGetUniformLocation(m_DirLightInjectionShader->GetProgram(), s_VoxelChannelNames[VoxelChannel::TexVoxelRadiance]);
        if (loc != -1)
            glBindImageTexture(BINDING_POINT_START_VOXEL_IMG + VoxelChannel::TexVoxelRadiance, m_VoxelizeTex[VoxelChannel::TexVoxelRadiance]->GetTexObj(), 0, GL_TRUE, 0, GL_READ_WRITE, m_VoxelizeTex[VoxelChannel::TexVoxelRadiance]->GetInternalFormat());
        loc = glGetUniformLocation(m_DirLightInjectionShader->GetProgram(), "depthMap");
        if (loc != -1)
        {
            glActiveTexture(GL_TEXTURE0);
            glBindTexture(GL_TEXTURE_2D, m_DirectionalDepthMap[i]);
			glUniform1i(loc, 0);
        }

        loc = glGetUniformLocation(m_DirLightInjectionShader->GetProgram(), "CurrDirLight.lightMtx");
        if (loc != -1)
            glProgramUniformMatrix4fv(m_DirLightInjectionShader->GetProgram(), loc, 1, false, glm::value_ptr(dir_light.m_LightMtx));

        loc = glGetUniformLocation(m_DirLightInjectionShader->GetProgram(), "LightProjMtx");
        if (loc != -1)
            glProgramUniformMatrix4fv(m_DirLightInjectionShader->GetProgram(), loc, 1, false, glm::value_ptr(light_proj_mtx));

        loc = glGetUniformLocation(m_DirLightInjectionShader->GetProgram(), "CurrDirLight.direction");
        if(loc != -1)
        glProgramUniform4f(m_DirLightInjectionShader->GetProgram(), loc, 
            dir_light.m_Direction.x,
            dir_light.m_Direction.y,
            dir_light.m_Direction.z,
            1.0f);

        loc = glGetUniformLocation(m_DirLightInjectionShader->GetProgram(), "CurrDirLight.color");
        if(loc != -1)
        glProgramUniform4f(m_DirLightInjectionShader->GetProgram(), loc, 
            dir_light.m_Color.x, 
            dir_light.m_Color.y, 
            dir_light.m_Color.z, 
            dir_light.m_Color.w);

        m_VoxelCam->UpdateCamMtx(UniformBufferBinding::kVoxelSpaceReconstruct);
        
        loc = glGetUniformLocation(m_DirLightInjectionShader->GetProgram(), "CurrLightPass");
        if (loc != -1)
            glProgramUniform1ui(m_DirLightInjectionShader->GetProgram(), loc, i + 1); //light pass

        glDispatchCompute(
            (m_DirLightInjectionRes + m_LightInjectionGroupSize - 1) / m_LightInjectionGroupSize,
            (m_DirLightInjectionRes + m_LightInjectionGroupSize - 1) / m_LightInjectionGroupSize, 1);

        glMemoryBarrier(GL_SHADER_IMAGE_ACCESS_BARRIER_BIT);
	}
	
 	EnableShadowSampling();

    glBindFramebuffer(GL_FRAMEBUFFER, 0);
	glEnable(GL_DEPTH_TEST);
	glEnable(GL_CULL_FACE);
	glColorMask(GL_TRUE, GL_TRUE, GL_TRUE, GL_TRUE);
	glDepthMask(GL_TRUE);
    uint32_t vw, vh;
    World::GetInst().GetViewportSize(vw, vh);
	glViewport(0, 0, vw, vh);
}

void VoxelizeController::DispatchPointLightInjection()
{
    for (uint32_t i = 0; i < VoxelChannel::Count; i++)
        if (m_VoxelizeTex[i] == nullptr) return;

	const LightManager& light_manager = World::GetInst().GetLightManager();
	const RendererList& renderers = World::GetInst().GetRenderers();

    glEnable(GL_DEPTH_TEST);
    glEnable(GL_CULL_FACE);
    glPolygonMode(GL_FRONT_AND_BACK, GL_FILL);
    glColorMask(GL_TRUE, GL_TRUE, GL_TRUE, GL_TRUE);
    glDepthMask(GL_TRUE);
    glViewport(0, 0, m_PointLightInjectionRes, m_PointLightInjectionRes);

    for (uint32_t i = 0; i < light_manager.GetPointLightCount(); i++)
    {
		glBindFramebuffer(GL_FRAMEBUFFER, m_DepthFBO);
		glFramebufferTexture(GL_FRAMEBUFFER, GL_DEPTH_ATTACHMENT, m_CubeDepthMap[i], 0);
		glDrawBuffer(GL_NONE);
		glReadBuffer(GL_NONE);
#if _DEBUG
		GLenum status = glCheckFramebufferStatus(GL_FRAMEBUFFER);
		assert(status == GL_FRAMEBUFFER_COMPLETE);
#endif


        const PointLight& point_light = light_manager.GetPointLight(i);

        //overwrite main camera
        glm::mat4x4 light_mtx[6];
		glm::mat4x4 light_proj_mtx;
        const float atten_cutoff = 0.05f;
		const float range_far = light_manager.ComputePointLightCutoffRadius(point_light, atten_cutoff);
		glm::vec2 capture_range(0.1f, range_far);
		//glm::vec2 capture_range(0.1f, 10000.0f);
        point_light.GomputeCubeLightMtx(capture_range.x, capture_range.y, light_mtx, light_proj_mtx);

        // example cube map coordinate retrieve
		//save
		//float n = capture_range.x, f = capture_range.y;
		//auto a = light_proj_mtx * light_mtx[3] * glm::vec4(348, -290, -67.3, 1);
		//auto v = light_mtx[3] * glm::vec4(348, -290, -67.3, 1);
		//auto b = a / a.w;
		//float d = a.z / a.w * 0.5 + 0.5;

		//float e = d * 2.0 - 1.0;
		//float z = ((n - f) * e - 2 * n * f) / (n + f);
		//z = -v.z;
		//float linear = (z - n) / (f - n);
		//float frag_x = b.x * 0.5 + 0.5;
		//float frag_y = b.y * 0.5 + 0.5;

		////restore a (input: frag_x, frag_y and linear)
		//float z_r = n + (f - n) * linear;
		//float x_r = frag_x * 2.0 - 1.0;
		//float y_r = frag_y * 2.0 - 1.0;
		//x_r *= z_r;
		//y_r *= z_r;
		//z_r = -z_r;
		//float w = -z_r;
		//z_r = (n + f + 2 * n * f / z_r) / (f - n);
		//z_r *= w;

		//glm::vec4 o = glm::inverse(light_proj_mtx * light_mtx[3]) * glm::vec4(x_r, y_r, z_r, w);


        glBindBufferRange(GL_UNIFORM_BUFFER, (uint8_t)UniformBufferBinding::kPointLightCapture, m_PointLightCaptureUBuffer, 0, GetPointLightCaptureUBufferSize());
        glBindBuffer(GL_UNIFORM_BUFFER, m_PointLightCaptureUBuffer);

        glBufferSubData(GL_UNIFORM_BUFFER, 0, 6 * sizeof(glm::mat4), light_mtx);
		glBufferSubData(GL_UNIFORM_BUFFER, 6 * sizeof(glm::mat4), sizeof(glm::mat4), glm::value_ptr(light_proj_mtx));
        glBufferSubData(GL_UNIFORM_BUFFER, 7 * sizeof(glm::mat4), sizeof(glm::vec4), glm::value_ptr(point_light.m_Position));
		glBufferSubData(GL_UNIFORM_BUFFER, 7 * sizeof(glm::mat4) + sizeof(glm::vec4), sizeof(glm::vec2), glm::value_ptr(capture_range));
        
        glBindBuffer(GL_UNIFORM_BUFFER, 0);

        glClear(GL_DEPTH_BUFFER_BIT);
        for (ModelRenderer* renderer : renderers)
            renderer->Render(RenderPass::kPointLightInjection, point_light.m_Position, range_far);

        //now that we have depth information, do light injection
		glUseProgram(m_PointLightInjectionShader->GetProgram());

		GLint loc = glGetUniformLocation(m_PointLightInjectionShader->GetProgram(), s_VoxelChannelNames[VoxelChannel::TexVoxelAlbedo]);
		if (loc != -1)
			glBindImageTexture(BINDING_POINT_START_VOXEL_IMG + VoxelChannel::TexVoxelAlbedo, m_VoxelizeTex[VoxelChannel::TexVoxelAlbedo]->GetTexObj(), 0, GL_TRUE, 0, GL_READ_ONLY, m_VoxelizeTex[VoxelChannel::TexVoxelAlbedo]->GetInternalFormat());
		loc = glGetUniformLocation(m_PointLightInjectionShader->GetProgram(), s_VoxelChannelNames[VoxelChannel::TexVoxelNormal]);
		if (loc != -1)
			glBindImageTexture(BINDING_POINT_START_VOXEL_IMG + VoxelChannel::TexVoxelNormal, m_VoxelizeTex[VoxelChannel::TexVoxelNormal]->GetTexObj(), 0, GL_TRUE, 0, GL_READ_ONLY, m_VoxelizeTex[VoxelChannel::TexVoxelNormal]->GetInternalFormat());
		loc = glGetUniformLocation(m_PointLightInjectionShader->GetProgram(), s_VoxelChannelNames[VoxelChannel::TexVoxelRadiance]);
		if (loc != -1)
			glBindImageTexture(BINDING_POINT_START_VOXEL_IMG + VoxelChannel::TexVoxelRadiance, m_VoxelizeTex[VoxelChannel::TexVoxelRadiance]->GetTexObj(), 0, GL_TRUE, 0, GL_READ_WRITE, m_VoxelizeTex[VoxelChannel::TexVoxelRadiance]->GetInternalFormat());
		loc = glGetUniformLocation(m_PointLightInjectionShader->GetProgram(), "cubeDepthMap");
		if (loc != -1)
		{
			glActiveTexture(GL_TEXTURE0);
			glBindTexture(GL_TEXTURE_CUBE_MAP, m_CubeDepthMap[i]);
			glUniform1i(loc, 0);
		}

		m_VoxelCam->UpdateCamMtx(UniformBufferBinding::kVoxelSpaceReconstruct);
		//we've already updated light capture buffer before
		
		//assume light info has been updated by light manager before this get called

		loc = glGetUniformLocation(m_PointLightInjectionShader->GetProgram(), "CurrPointLight.position");
		if(loc != -1)
			glProgramUniform4f(m_PointLightInjectionShader->GetProgram(), loc,
				point_light.m_Position.x,
				point_light.m_Position.y,
				point_light.m_Position.z,
				1.0f);

		loc = glGetUniformLocation(m_PointLightInjectionShader->GetProgram(), "CurrPointLight.color");
		if (loc != -1)
			glProgramUniform4f(m_PointLightInjectionShader->GetProgram(), loc,
				point_light.m_Color.x,
				point_light.m_Color.y,
				point_light.m_Color.z,
				point_light.m_Color.w);

        loc = glGetUniformLocation(m_PointLightInjectionShader->GetProgram(), "CurrLightPass");
        if (loc != -1)
            glProgramUniform1ui(m_PointLightInjectionShader->GetProgram(), loc, i + 1 + light_manager.GetDirLightCount()); //light pass

		glDispatchCompute(
			(m_PointLightInjectionRes + m_LightInjectionGroupSize - 1) / m_LightInjectionGroupSize,
			(m_PointLightInjectionRes + m_LightInjectionGroupSize - 1) / m_LightInjectionGroupSize, 1);

		glMemoryBarrier(GL_SHADER_IMAGE_ACCESS_BARRIER_BIT);
    }

    glBindFramebuffer(GL_FRAMEBUFFER, 0);
    glEnable(GL_DEPTH_TEST);
    glEnable(GL_CULL_FACE);
    glColorMask(GL_TRUE, GL_TRUE, GL_TRUE, GL_TRUE);
    glDepthMask(GL_TRUE);
    uint32_t vw, vh;
    World::GetInst().GetViewportSize(vw, vh);
    glViewport(0, 0, vw, vh);
}

void VoxelizeController::MipmapRadiance()
{
	glUseProgram(m_AnisotropicMipmapShaderLeaf->GetProgram());
    glActiveTexture(GL_TEXTURE0);
    glBindTexture(GL_TEXTURE_3D, m_VoxelizeTex[VoxelChannel::TexVoxelRadiance]->GetTexObj());
	//glBindImageTexture(0, m_VoxelizeTex[VoxelChannel::TexVoxelRadiance]->GetTexObj(), 0, GL_TRUE, 0, GL_READ_ONLY, m_VoxelizeTex[VoxelChannel::TexVoxelRadiance]->GetInternalFormat());
	for (uint32_t i = 0; i < s_AnisotropicMipmapCount; i++)
		glBindImageTexture(i + 1, m_AnisoRadianceMipmap[i]->GetTexObj(), 0, GL_TRUE, 0, GL_WRITE_ONLY, m_AnisoRadianceMipmap[i]->GetInternalFormat());

	glDispatchCompute(
		(m_VoxelDim / 2 + (m_AnisoMipmapGroupSize - 1)) / (m_AnisoMipmapGroupSize),
		(m_VoxelDim / 2 + (m_AnisoMipmapGroupSize - 1)) / (m_AnisoMipmapGroupSize),
		(m_VoxelDim / 2 + (m_AnisoMipmapGroupSize - 1)) / (m_AnisoMipmapGroupSize));

	glMemoryBarrier(GL_SHADER_IMAGE_ACCESS_BARRIER_BIT);

    //GLint p[3];
    //glGetIntegeri_v(GL_MAX_COMPUTE_WORK_GROUP_SIZE, 0, p);
    //glGetIntegeri_v(GL_MAX_COMPUTE_WORK_GROUP_SIZE, 1, p + 1);
    //glGetIntegeri_v(GL_MAX_COMPUTE_WORK_GROUP_SIZE, 2, p + 2);

    //glGetIntegeri_v(GL_MAX_COMPUTE_WORK_GROUP_COUNT, 0, p);
    //glGetIntegeri_v(GL_MAX_COMPUTE_WORK_GROUP_COUNT, 1, p + 1);
    //glGetIntegeri_v(GL_MAX_COMPUTE_WORK_GROUP_COUNT, 2, p + 2);

    //glGetIntegerv(GL_MAX_COMPUTE_WORK_GROUP_INVOCATIONS, p);


	uint32_t level_count = 1;
	uint32_t dim = m_VoxelDim / 2;
	while (dim >>= 1) ++level_count;
	for (uint32_t i = 0; i < s_AnisotropicMipmapCount; i++)
	{
        glUseProgram(m_AnisotropicMipmapShaderInterior[i]->GetProgram());
		dim = m_VoxelDim / 4;
		for (uint32_t level = 0; level < level_count - 1; level++)
		{
            glActiveTexture(GL_TEXTURE0);
            glBindTexture(GL_TEXTURE_3D, m_AnisoRadianceMipmap[i]->GetTexObj());

            GLint loc = glGetUniformLocation(m_AnisotropicMipmapShaderInterior[i]->GetProgram(), "ChildMipLevel");
            glProgramUniform1i(m_AnisotropicMipmapShaderInterior[i]->GetProgram(), loc, level);
			//glBindImageTexture(0, m_AnisoRadianceMipmap[i]->GetTexObj(), level, GL_TRUE, 0, GL_READ_ONLY, m_AnisoRadianceMipmap[i]->GetInternalFormat());
			glBindImageTexture(1, m_AnisoRadianceMipmap[i]->GetTexObj(), level + 1, GL_TRUE, 0, GL_WRITE_ONLY, m_AnisoRadianceMipmap[i]->GetInternalFormat());
			glDispatchCompute(
				(dim + (m_BoxMipmapGroupSize - 1)) / (m_BoxMipmapGroupSize),
				(dim + (m_BoxMipmapGroupSize - 1)) / (m_BoxMipmapGroupSize),
				(dim + (m_BoxMipmapGroupSize - 1)) / (m_BoxMipmapGroupSize));

			glMemoryBarrier(GL_SHADER_IMAGE_ACCESS_BARRIER_BIT);
		}
	}

}

void VoxelizeController::LightSpaceBBox(const DirLight& light, glm::vec3& min, glm::vec3& max) const
{
    min.x = min.y = min.z = FLT_MAX;
    max.x = max.y = max.z = FLT_MIN;

    glm::vec4 center4(m_Center.x, m_Center.y, m_Center.z, 0.0f);

    glm::vec4 bbox[8]
    {
        center4 + glm::vec4{ -m_Extent.x, -m_Extent.y, -m_Extent.z, 1.0f },
        center4 + glm::vec4{ -m_Extent.x, -m_Extent.y, m_Extent.z, 1.0f },
        center4 + glm::vec4{ -m_Extent.x, m_Extent.y, -m_Extent.z, 1.0f },
        center4 + glm::vec4{ -m_Extent.x, m_Extent.y, m_Extent.z, 1.0f },

        center4 + glm::vec4{ m_Extent.x, -m_Extent.y, -m_Extent.z, 1.0f },
        center4 + glm::vec4{ m_Extent.x, -m_Extent.y, m_Extent.z, 1.0f },
        center4 + glm::vec4{ m_Extent.x, m_Extent.y, -m_Extent.z, 1.0f },
        center4 + glm::vec4{ m_Extent.x, m_Extent.y, m_Extent.z, 1.0f },
    };

    for (uint32_t i = 0; i < 8; i++)
    {
        glm::vec4 curr = light.m_LightMtx * bbox[i];
        if (curr.x < min.x)
            min.x = curr.x;
        if (curr.x > max.x)
            max.x = curr.x;

        if (curr.y < min.y)
            min.y = curr.y;
        if (curr.y > max.y)
            max.y = curr.y;

        if (curr.z < min.z)
            min.z = curr.z;
        if (curr.z > max.z)
            max.z = curr.z;
    }
}
