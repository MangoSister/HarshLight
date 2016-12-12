#include "World.h"

#include <cassert>
#include <cuda_runtime.h>
#include <cuda_gl_interop.h>
#include <device_launch_parameters.h>

#include "VoxelInvert.h"

void World::MouseCallback(GLFWwindow * window, double xpos, double ypos)
{
	World& world = World::GetInst();
	static double s_LastMouseX = 0.5 * static_cast<double>(world.m_FullRenderWidth);
	static double s_LastMouseY = 0.5 * static_cast<double>(world.m_FullRenderHeight);

	double xoffset = xpos - s_LastMouseX;
	double yoffset = s_LastMouseY - ypos;
	s_LastMouseX = xpos;
	s_LastMouseY = ypos;

	xoffset *= world.m_MouseSensitivity;
	yoffset *= world.m_MouseSensitivity;

	static float yaw = 0.0f;
	static float pitch = 0.0f;
    const float MAX_PITCH = 89.0f;
	//printf("%f %f\n", yaw, pitch);
	yaw += static_cast<float>(xoffset);
	yaw = fmod(yaw, 360.0f);  
	pitch += static_cast<float>(yoffset);
    pitch = std::fmin(std::fmax(pitch, -MAX_PITCH), MAX_PITCH);

	Camera* main_camera = world.GetMainCamera();
	if (main_camera)
	{
		glm::vec3 forward;
		forward.z = std::cosf(glm::radians(pitch)) * std::cosf(glm::radians(yaw));
		forward.y = std::sinf(glm::radians(pitch));
		forward.x = std::cosf(glm::radians(pitch)) * std::sinf(glm::radians(yaw));
		forward = normalize(forward);
		if (std::fabsf(glm::dot(forward, glm::vec3(0.0f, 1.0f, 0.0f))) != 1.0f)
			main_camera->LookAtDir(forward, glm::vec3(0.0f, 1.0f, 0.0f));
		else
			main_camera->LookAtDir(forward, glm::vec3(0.0f, 0.0f, forward.y > 0.0f ? -1.0f : 1.0f));
	}
}

void World::KeyboardCallback(GLFWwindow * window, int key, int scancode, int action, int mods)
{
    KeyStatusMap& map = World::GetInst().m_KeyStatusMap;
    auto key_it = map.find(key);
    if (key_it != map.cend())
        key_it->second = action;
    else
        map.insert(std::make_pair(key, action));
}

void World::ComputeGeometryPass()
{
    glClearColor(0.0f, 0.0f, 0.0f, 0.0f);
    glBindFramebuffer(GL_FRAMEBUFFER, m_GBufferFBO);
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
    glEnable(GL_DEPTH_TEST);
    glEnable(GL_CULL_FACE);
    glDisable(GL_BLEND);
    glViewport(0, 0, m_FullRenderWidth, m_FullRenderHeight);

    if (m_MainCamera)
        m_MainCamera->UpdateCamMtx(UniformBufferBinding::kMainCam);
    else
        fprintf(stderr, "WARNING: MainCamera is null\n");

    for (ModelRenderer* renderer : m_Renderers)
        renderer->Render(RenderPass::kGeometry);


    glBindFramebuffer(GL_FRAMEBUFFER, 0);
}

void World::ComputeShadingPass()
{
    glBindFramebuffer(GL_FRAMEBUFFER, 0);
    glClearColor(0.2f, 0.3f, 0.5f, 1.0f);
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
	glViewport(0, 0, m_FullRenderWidth, m_FullRenderHeight);

    if (m_MainCamera)
        m_MainCamera->UpdateCamMtx(UniformBufferBinding::kMainCam);
    else
        fprintf(stderr, "WARNING: MainCamera is null\n");

    if (m_VoxelizeCamera)
        m_VoxelizeCamera->UpdateCamMtx(UniformBufferBinding::kVoxelSpaceReconstruct);
    else
        fprintf(stderr, "WARNING: VoxelizeCamera is null\n");

	//render quad
    m_DeferredShadingQuad->Render(RenderPass::kDeferredIndirectDiffuse);
   
}

void World::ComputeIndirectDiffusePass()
{
	glBindFramebuffer(GL_FRAMEBUFFER, m_IndirectDiffuseFBO);
	glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, m_IndirectDiffuseHalfBuffer, 0);

	glClearColor(0.8f, 0.3f, 0.2f, 1.0f);
	glClear(GL_COLOR_BUFFER_BIT); //not using stencil buffer?

	const uint32_t half_render_width = m_FullRenderWidth / 2;
	const uint32_t half_render_height = m_FullRenderHeight / 2;
	glViewport(0, 0, half_render_width, half_render_height);

	if (m_MainCamera)
		m_MainCamera->UpdateCamMtx(UniformBufferBinding::kMainCam);
	else
		fprintf(stderr, "WARNING: MainCamera is null\n");

	if (m_VoxelizeCamera)
		m_VoxelizeCamera->UpdateCamMtx(UniformBufferBinding::kVoxelSpaceReconstruct);
	else
		fprintf(stderr, "WARNING: VoxelizeCamera is null\n");

	m_DeferredShadingQuad->Render(RenderPass::kDeferredIndirectDiffuse);

	glBindFramebuffer(GL_FRAMEBUFFER, 0);

}

void World::ComputeFinalCompositionPass()
{
	glBindFramebuffer(GL_FRAMEBUFFER, 0);
	glClearColor(0.2f, 0.3f, 0.5f, 1.0f);
	glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
	glViewport(0, 0, m_FullRenderWidth, m_FullRenderHeight);

	if (m_MainCamera)
		m_MainCamera->UpdateCamMtx(UniformBufferBinding::kMainCam);
	else
		fprintf(stderr, "WARNING: MainCamera is null\n");

	if (m_VoxelizeCamera)
		m_VoxelizeCamera->UpdateCamMtx(UniformBufferBinding::kVoxelSpaceReconstruct);
	else
		fprintf(stderr, "WARNING: VoxelizeCamera is null\n");

	//render quad
	m_DeferredShadingQuad->Render(RenderPass::kDeferredFinalComposition);
}

void World::RenderUIText()
{
	glEnable(GL_BLEND);
	glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
	glDisable(GL_DEPTH_TEST);
	glDisable(GL_CULL_FACE);
	static char char_buf[100];

	memset(char_buf, 0, 100);
	sprintf(char_buf, "Free Move: W/S/A/D/Q/E and Mouse");
	m_TextManager.RenderText(std::string(char_buf), 25.0f, 120.0f, 0.5f, glm::vec3(0.5, 0.8f, 0.2f));

	memset(char_buf, 0, 100);
	sprintf(char_buf, "Rotate Main Light: I/J/K/L");
	m_TextManager.RenderText(std::string(char_buf), 25.0f, 90.0f, 0.5f, glm::vec3(0.5, 0.8f, 0.2f));

	memset(char_buf, 0, 100);
	sprintf(char_buf, "Toggle Secondary Light: F");
	m_TextManager.RenderText(std::string(char_buf), 25.0f, 60.0f, 0.5f, glm::vec3(0.5, 0.8f, 0.2f));

	memset(char_buf, 0, 100);
	sprintf(char_buf, "FPS: %.2f", 1.0f / m_CurrDeltaTime); 	//fps: 1 / elapsed
	m_TextManager.RenderText(std::string(char_buf), 25.0f, 30.0f, 0.5f, glm::vec3(0.5, 0.8f, 0.2f));
}

void World::SetWindow(GLFWwindow* window, uint32_t width, uint32_t height)
{
    m_Window = window;
    glfwSetKeyCallback(window, KeyboardCallback);
	glfwSetInputMode(window, GLFW_CURSOR, GLFW_CURSOR_DISABLED);
	glfwSetCursorPosCallback(window, MouseCallback);
	m_FullRenderWidth = width;
	m_FullRenderHeight = height;
}

const ActorList& World::GetActors() const
{
    return m_Actors;
}

void World::RegisterActor(Actor* actor)
{
#ifdef _DEBUG
    assert(actor != nullptr);
#endif

    auto comps = actor->GetAllComponents();
    for (Component* comp : comps)
    {
        m_Components.push_back(comp);
    }
    auto renderers = actor->GetAllRenderers();
    for (ModelRenderer* renderer : renderers)
    {
        m_Renderers.push_back(renderer);
        FrameBufferDisplay* display = dynamic_cast<FrameBufferDisplay*>(renderer);
        if (display)
            m_FrameBufferDisplays.push_back(display);
    }


    m_Actors.push_back(actor);

}

const ModelList& World::GetModels() const
{
	return m_Models;
}

void World::RegisterModel(Model* model)
{
#if _DEBUG
	assert(model != nullptr);
#endif
	m_Models.push_back(model);
}

const ShaderList & World::GetShaders() const
{
	return m_Shaders;
}

void World::RegisterShader(ShaderProgram * shader)
{
#ifdef _DEBUG
	assert(shader != nullptr);
#endif
	m_Shaders.push_back(shader);
}

const MaterialList& World::GetMaterials() const
{
    return m_Materials;
}

void World::RegisterMaterial(Material* material)
{
#ifdef _DEBUG
    assert(material != nullptr);
#endif
    m_Materials.push_back(material);
}

const Texture2dDict& World::GetTexture2ds() const
{
    return m_Textures2d;
}

void World::RegisterTexture2d(const std::string& path, Texture2d* tex2d)
{
#ifdef _DEBUG
    assert(tex2d);
    assert(m_Textures2d.find(path) == m_Textures2d.cend()); //no repeat registration!
#endif
    m_Textures2d.insert({ path, tex2d });
}

const Texture3dList& World::GetTexture3ds() const
{
	return m_Textures3d;
}

void World::RegisterTexture3d(Texture3dCompute* tex3d)
{
#ifdef _DEBUG
	assert(tex3d);
#endif

	m_Textures3d.push_back(tex3d);
}

Camera* World::GetVoxelCamera() const
{
	return m_VoxelizeCamera;
}

void World::SetVoxelCamera(Camera * camera)
{
#ifdef _DEBUG
	assert(camera != nullptr);
#endif
	m_VoxelizeCamera = camera;
}

void World::SetMainCamera(Camera * camera)
{
#ifdef _DEBUG
	assert(camera != nullptr);
#endif
	m_MainCamera = camera;
}

Camera* World::GetMainCamera() const
{
	return m_MainCamera;
}

void World::SetMouseSensitivity(float sensitivity)
{
	m_MouseSensitivity = sensitivity;
}

const RendererList& World::GetRenderers() const
{
	return m_Renderers;
}

LightManager & World::GetLightManager()
{
    return m_LightManager;
}

const void World::GetFullRenderSize(uint32_t & width, uint32_t & height) const
{
	width = m_FullRenderWidth;
	height = m_FullRenderHeight;
}

void World::Start()
{
    /*----------------  Initialize G-buffer --------------*/
    glGenFramebuffers(1, &m_GBufferFBO);
    glBindFramebuffer(GL_FRAMEBUFFER, m_GBufferFBO);

    glGenTextures(1, &m_GPositionAndSpecPower);
    glBindTexture(GL_TEXTURE_2D, m_GPositionAndSpecPower);
    //RGBA 16F HERE 
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA16F, m_FullRenderWidth, m_FullRenderHeight, 0, GL_RGBA, GL_FLOAT, nullptr);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
    glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, m_GPositionAndSpecPower, 0);

    //RGBA 16F HERE 
    glGenTextures(1, &m_GNormalAndTangent);
    glBindTexture(GL_TEXTURE_2D, m_GNormalAndTangent);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA16F, m_FullRenderWidth, m_FullRenderHeight, 0, GL_RGBA, GL_FLOAT, nullptr);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
    glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT1, GL_TEXTURE_2D, m_GNormalAndTangent, 0);

    //RGBA 8 HERE
    glGenTextures(1, &m_GAlbedoAndSpecIntensity);
    glBindTexture(GL_TEXTURE_2D, m_GAlbedoAndSpecIntensity);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, m_FullRenderWidth, m_FullRenderHeight, 0, GL_RGBA, GL_UNSIGNED_BYTE, nullptr);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
    glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT2, GL_TEXTURE_2D, m_GAlbedoAndSpecIntensity, 0);

    GLuint attachments[3] = { GL_COLOR_ATTACHMENT0, GL_COLOR_ATTACHMENT1, GL_COLOR_ATTACHMENT2 };
    glDrawBuffers(3, attachments);

    //also depth buffer
    glGenRenderbuffers(1, &m_GDepthRBO);
    glBindRenderbuffer(GL_RENDERBUFFER, m_GDepthRBO);
    glRenderbufferStorage(GL_RENDERBUFFER, GL_DEPTH_COMPONENT, m_FullRenderWidth, m_FullRenderHeight);
    glFramebufferRenderbuffer(GL_FRAMEBUFFER, GL_DEPTH_ATTACHMENT, GL_RENDERBUFFER, m_GDepthRBO);

#if _DEBUG
	{
		GLenum status = glCheckFramebufferStatus(GL_FRAMEBUFFER);
		assert(status == GL_FRAMEBUFFER_COMPLETE);
	}
#endif

    glBindFramebuffer(GL_FRAMEBUFFER, 0);

	//half indirect diffuse buffer
	glGenFramebuffers(1, &m_IndirectDiffuseFBO);
	glBindFramebuffer(GL_FRAMEBUFFER, m_IndirectDiffuseFBO);

	const uint32_t half_render_width = m_FullRenderWidth / 2;
	const uint32_t half_render_height = m_FullRenderHeight / 2;
	glGenTextures(1, &m_IndirectDiffuseHalfBuffer);
	glBindTexture(GL_TEXTURE_2D, m_IndirectDiffuseHalfBuffer);
	//RGB 32 FLOAT HERE, we need high resolution for indirect diffuse color
	glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB32F, half_render_width, half_render_height, 0, GL_RGB, GL_FLOAT, nullptr);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
	glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, m_IndirectDiffuseHalfBuffer, 0);
	glBindTexture(GL_TEXTURE_2D, 0);
	
#if _DEBUG
	{
		GLenum status = glCheckFramebufferStatus(GL_FRAMEBUFFER);
		assert(status == GL_FRAMEBUFFER_COMPLETE);
	}
#endif

	glBindFramebuffer(GL_FRAMEBUFFER, 0);

	ShaderProgram* ds_indirect_diffuse_shader = new ShaderProgram();
	ds_indirect_diffuse_shader->AddVertShader("src/shaders/ds_shading_vert.glsl");
	ds_indirect_diffuse_shader->AddFragShader("src/shaders/ds_indirect_diffuse_frag.glsl");
	ds_indirect_diffuse_shader->LinkProgram();
	RegisterShader(ds_indirect_diffuse_shader);

	ShaderProgram* ds_final_composition = new ShaderProgram();
	ds_final_composition->AddVertShader("src/shaders/ds_shading_vert.glsl");
	ds_final_composition->AddFragShader("src/shaders/ds_final_composition_frag.glsl");
	ds_final_composition->LinkProgram();
	RegisterShader(ds_final_composition);

    Model* quad = new Model(Model::Primitive::kQuad);
    RegisterModel(quad);

    m_DeferredShadingQuad = new ModelRenderer(quad);
    m_DeferredShadingQuad->SetRenderPass(RenderPass::kDeferredIndirectDiffuse | RenderPass::kDeferredFinalComposition);
    m_DeferredShadingQuad->MoveTo({ 0.0f, 0.0f, 0.0f });
    m_DeferredShadingQuad->ScaleTo({ 2.0f, 2.0f, 1.0f });

	{
		Material* mat_ds_indirect_diffuse = new Material();
		mat_ds_indirect_diffuse->SetShader(ds_indirect_diffuse_shader);
		mat_ds_indirect_diffuse->SetFloatParam("VoxelDim", static_cast<float>(m_VoxelizeController->GetVoxelDim()));
		mat_ds_indirect_diffuse->SetFloatParam("VoxelScale", m_VoxelizeController->GetVoxelScale());
		mat_ds_indirect_diffuse->AddTexture2dDirect(m_GPositionAndSpecPower, "GPositionAndSpecPower");
		mat_ds_indirect_diffuse->AddTexture2dDirect(m_GNormalAndTangent, "GNormalAndTangent");
		mat_ds_indirect_diffuse->AddTexture2dDirect(m_GAlbedoAndSpecIntensity, "GAlbedoAndSpecIntensity");
		mat_ds_indirect_diffuse->AddTexture(m_VoxelizeController->GetVoxelizeTex(VoxelChannel::TexVoxelRadiance), "ImgRadianceLeaf", TexUsage::kRegularTexture, 0, 0);
		char sampler_name[30];
		for (uint32_t i = 0; i < 6; i++)
		{
			memset(sampler_name, 0, 30);
			sprintf(sampler_name, "ImgRadianceInterior[%d]", i);
			mat_ds_indirect_diffuse->AddTexture(m_VoxelizeController->GetAnisoRadianceMipmap(i), sampler_name, TexUsage::kRegularTexture, 0, 0);
		}
		RegisterMaterial(mat_ds_indirect_diffuse);

		m_DeferredShadingQuad->AddMaterial(RenderPass::kDeferredIndirectDiffuse, mat_ds_indirect_diffuse);
	}

	{
		Material* mat_ds_final_composition = new Material();
		mat_ds_final_composition->SetShader(ds_final_composition);
		mat_ds_final_composition->SetFloatParam("VoxelDim", static_cast<float>(m_VoxelizeController->GetVoxelDim()));
		mat_ds_final_composition->SetFloatParam("VoxelScale", m_VoxelizeController->GetVoxelScale());
		mat_ds_final_composition->AddTexture2dDirect(m_GPositionAndSpecPower, "GPositionAndSpecPower");
		mat_ds_final_composition->AddTexture2dDirect(m_GNormalAndTangent, "GNormalAndTangent");
		mat_ds_final_composition->AddTexture2dDirect(m_GAlbedoAndSpecIntensity, "GAlbedoAndSpecIntensity");
		mat_ds_final_composition->AddTexture2dDirect(m_IndirectDiffuseHalfBuffer, "BufIndirectDiffuse");
		mat_ds_final_composition->AddTexture(m_VoxelizeController->GetVoxelizeTex(VoxelChannel::TexVoxelRadiance), "ImgRadianceLeaf", TexUsage::kRegularTexture, 0, 0);
		char sampler_name[30];
		for (uint32_t i = 0; i < 6; i++)
		{
			memset(sampler_name, 0, 30);
			sprintf(sampler_name, "ImgRadianceInterior[%d]", i);
			mat_ds_final_composition->AddTexture(m_VoxelizeController->GetAnisoRadianceMipmap(i), sampler_name, TexUsage::kRegularTexture, 0, 0);
		}
		for (uint32_t i = 0; i < LightManager::s_DirLightMaxNum; i++)
		{
			memset(sampler_name, 0, 30);
			sprintf(sampler_name, "TexDirShadow[%u]", i);
			mat_ds_final_composition->AddTexture2dDirect(m_VoxelizeController->GetDirectionalDepthMap(i), sampler_name);
		}
		RegisterMaterial(mat_ds_final_composition);

		m_DeferredShadingQuad->AddMaterial(RenderPass::kDeferredFinalComposition, mat_ds_final_composition);
	}


    m_TextManager.Init();

    for (Component* comp : m_Components)
    {
#ifdef _DEBUG
        assert(comp != nullptr);
#endif
        comp->Start();
    }

    m_LastTime = std::chrono::system_clock::now();
    m_CurrTime = std::chrono::system_clock::now();

    /*--------- pass 0: voxelize scene ---------*/
	//m_VoxelizeController->DispatchVoxelization();
	//is executed in its Start() function

    //run a testing kernel
	//cudaSurfaceObject_t surf_objs[VoxelChannel::Count];
 //   m_VoxelizeController->TransferVoxelDataToCuda(surf_objs);
 //   LaunchKernelVoxelInvert(m_VoxelizeController->GetVoxelDim(), surf_objs[0]);
 //   m_VoxelizeController->FinishVoxelDataFromCuda(surf_objs);
}

void World::MainLoop()
{
	m_CurrTime = std::chrono::system_clock::now();
	m_CurrDeltaTime = static_cast<float>((std::chrono::duration<double>(m_CurrTime - m_LastTime)).count());
	m_LastTime = m_CurrTime;

	m_LightManager.UpdateLight(UniformBufferBinding::kLight);
    /*--------- CPU update ---------*/
    for (Component* comp : m_Components)
    {
        assert(comp != nullptr);
        comp->Update(m_CurrDeltaTime);
    }
	/*--------- pass 1: light injection ---------*/
	//m_VoxelizeController->DispatchLightInjection();
	//is executed in its Update() function

    if (GetKey(GLFW_KEY_Z) == GLFW_PRESS)
        m_ToggleUI = !m_ToggleUI;

    glBindFramebuffer(GL_FRAMEBUFFER, 0);
    glClearColor(0.2f, 0.3f, 0.5f, 1.0f);
   // glClearColor(0.0f, 0.0f, 1.0f, 1.0f);
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
    glEnable(GL_DEPTH_TEST);
	glEnable(GL_CULL_FACE);
    glViewport(0, 0, m_FullRenderWidth, m_FullRenderHeight);

    ComputeGeometryPass();
    ComputeIndirectDiffusePass();
	ComputeFinalCompositionPass();

    if (m_ToggleUI)
    {
        RenderUIText();
    }
 
    //maintain 3-status key map
    for (auto& it : m_KeyStatusMap)
    {
        if (it.second == GLFW_PRESS)
            it.second = GLFW_REPEAT;
    }
}

std::vector<Material*> World::LoadDefaultMaterialsForModel(Model * model)
{
    std::vector<Material*> out;
    Assimp::Importer import;
    const aiScene* scene = import.ReadFile(model->GetRawPath(),
		aiProcess_FlipUVs | aiProcess_PreTransformVertices |
		aiProcess_FlipWindingOrder | // seems like models we use are all CW order...
		aiProcess_FindDegenerates |
		aiProcess_OptimizeMeshes |
		aiProcess_OptimizeGraph |
		aiProcess_JoinIdenticalVertices |
		aiProcess_CalcTangentSpace |
		aiProcess_GenSmoothNormals |
		aiProcess_Triangulate |
		aiProcess_FixInfacingNormals |
		aiProcess_FindInvalidData |
		aiProcess_ValidateDataStructure);

    if (!scene || scene->mFlags == AI_SCENE_FLAGS_INCOMPLETE || !scene->mRootNode)
    {
        fprintf(stderr, "assimp error: %s\n", import.GetErrorString());
        return out;
    }
    
    for (uint32_t i = 0; i < scene->mNumMaterials; i++)
    {
        Material* curr_material = new Material();
//#ifdef _DEBUG
//        assert(albedo_num > 0); //temporary
//#endif
		aiString fk = scene->mMaterials[i]->mProperties[0]->mKey;
        if (scene->mMaterials[i]->GetTextureCount(aiTextureType_DIFFUSE) > 0)
        {
            aiString albedo_path;
            scene->mMaterials[i]->GetTexture(aiTextureType::aiTextureType_DIFFUSE, 0, &albedo_path);
#ifdef _DEBUG
            assert(albedo_path.C_Str());
#endif


            std::string albedo_path_str(albedo_path.C_Str());
            for (size_t i = 0; i < albedo_path_str.length(); i++)
            {
                if (albedo_path_str[i] == '\\')
                    albedo_path_str[i] = '/';
            }
			
			const char* model_path = model->GetRawPath();

			//for (size_t i = 0; model_path[i] != '\0'; i++)
			//{
			//	if (model_path[i] == '\\')
			//		model_path[i] = '/';
			//}
            const char* directory = strrchr(model_path, '/');
            if (directory)
                albedo_path_str.insert(0, model_path, 0, directory - model_path + 1);

            Texture2d* curr_tex2d = nullptr;
            auto tex_iter = m_Textures2d.find(albedo_path_str);
            if (tex_iter == m_Textures2d.cend())
            {
                curr_tex2d = new Texture2d(albedo_path_str.c_str());
                RegisterTexture2d(albedo_path_str, curr_tex2d);
            }
            else
            {
                curr_tex2d = tex_iter->second;
            }

            curr_material->AddTexture(curr_tex2d, "TexAlbedo");
        }
		else
		{
			curr_material->AddTexture(m_DefaultBlackTex, "TexAlbedo");
		}

		if (scene->mMaterials[i]->GetTextureCount(aiTextureType_HEIGHT) > 0)
		{
			aiString normal_path;
			scene->mMaterials[i]->GetTexture(aiTextureType::aiTextureType_HEIGHT, 0, &normal_path);
#ifdef _DEBUG
			assert(normal_path.C_Str());
#endif

			std::string normal_path_str(normal_path.C_Str());
			for (size_t i = 0; i < normal_path_str.length(); i++)
			{
				if (normal_path_str[i] == '\\')
					normal_path_str[i] = '/';
			}

			const char* model_path = model->GetRawPath();

			//for (size_t i = 0; model_path[i] != '\0'; i++)
			//{
			//	if (model_path[i] == '\\')
			//		model_path[i] = '/';
			//}
			const char* directory = strrchr(model_path, '/');
			if (directory)
				normal_path_str.insert(0, model_path, 0, directory - model_path + 1);

			Texture2d* curr_tex2d = nullptr;
			auto tex_iter = m_Textures2d.find(normal_path_str);
			if (tex_iter == m_Textures2d.cend())
			{
				curr_tex2d = new Texture2d(normal_path_str.c_str());
				RegisterTexture2d(normal_path_str, curr_tex2d);
			}
			else
			{
				curr_tex2d = tex_iter->second;
			}

			curr_material->AddTexture(curr_tex2d, "TexNormal");
		}
		else
		{
			curr_material->AddTexture(m_DefaultNormalTex, "TexNormal");
		}

		if (scene->mMaterials[i]->GetTextureCount(aiTextureType_SPECULAR) > 0)
		{
			aiString spec_path;
			scene->mMaterials[i]->GetTexture(aiTextureType::aiTextureType_SPECULAR, 0, &spec_path);
#ifdef _DEBUG
			assert(spec_path.C_Str());
#endif

			std::string spec_path_str(spec_path.C_Str());
			for (size_t i = 0; i < spec_path_str.length(); i++)
			{
				if (spec_path_str[i] == '\\')
					spec_path_str[i] = '/';
			}

			const char* model_path = model->GetRawPath();

			//for (size_t i = 0; model_path[i] != '\0'; i++)
			//{
			//	if (model_path[i] == '\\')
			//		model_path[i] = '/';
			//}
			const char* directory = strrchr(model_path, '/');
			if (directory)
				spec_path_str.insert(0, model_path, 0, directory - model_path + 1);

			Texture2d* curr_tex2d = nullptr;
			auto tex_iter = m_Textures2d.find(spec_path_str);
			if (tex_iter == m_Textures2d.cend())
			{
				curr_tex2d = new Texture2d(spec_path_str.c_str());
				RegisterTexture2d(spec_path_str, curr_tex2d);
			}
			else
			{
				curr_tex2d = tex_iter->second;
			}

			curr_material->AddTexture(curr_tex2d, "TexSpecular");
		}
		else
		{
			curr_material->AddTexture(m_DefaultBlackTex, "TexSpecular");
		}

		if (scene->mMaterials[i]->GetTextureCount(aiTextureType_OPACITY) > 0)
		{
			aiString opacity_path;
			scene->mMaterials[i]->GetTexture(aiTextureType::aiTextureType_OPACITY, 0, &opacity_path);
#ifdef _DEBUG
			assert(opacity_path.C_Str());
#endif

			std::string opacity_path_str(opacity_path.C_Str());
			for (size_t i = 0; i < opacity_path_str.length(); i++)
			{
				if (opacity_path_str[i] == '\\')
					opacity_path_str[i] = '/';
			}

			const char* model_path = model->GetRawPath();

			//for (size_t i = 0; model_path[i] != '\0'; i++)
			//{
			//	if (model_path[i] == '\\')
			//		model_path[i] = '/';
			//}
			const char* directory = strrchr(model_path, '/');
			if (directory)
				opacity_path_str.insert(0, model_path, 0, directory - model_path + 1);

			Texture2d* curr_tex2d = nullptr;
			auto tex_iter = m_Textures2d.find(opacity_path_str);
			if (tex_iter == m_Textures2d.cend())
			{
				curr_tex2d = new Texture2d(opacity_path_str.c_str());
				RegisterTexture2d(opacity_path_str, curr_tex2d);
			}
			else
			{
				curr_tex2d = tex_iter->second;
			}

			curr_material->AddTexture(curr_tex2d, "TexOpacityMask");
		}
		else
		{
			curr_material->AddTexture(m_DefaultWhiteTex, "TexOpacityMask");
		}

        RegisterMaterial(curr_material);
        out.push_back(curr_material);
    }

	if (!scene->mNumMaterials)
	{
		Material* empty_material = new Material();
		RegisterMaterial(empty_material);
		out.push_back(empty_material);
	}

    return out;
}

void World::Destroy()
{
	for (Actor*& actor : m_Actors)
	{
		if (actor)
		{
			delete actor;
			actor = nullptr;
		}
	}

	for (Model*& model : m_Models)
	{
		if (model)
		{
			delete model;
			model = nullptr;
		}
	}

	for (Material*& material : m_Materials)
	{
		if (material)
		{
			delete material;
			material = nullptr;
		}
	}

	for (auto it = m_Textures2d.begin(); it != m_Textures2d.end(); it++)
	{
		//tex2d
		if (it->second)
		{
			delete (it->second);
			it->second = nullptr;
		}
	}

	for (Texture3dCompute*& tex3d : m_Textures3d)
	{
		if (tex3d)
		{
			delete tex3d;
			tex3d = nullptr;
		}
	}

	for (ShaderProgram*& shader : m_Shaders)
	{
		if (shader)
		{
			delete shader;
			shader = nullptr;
		}
	}

    /*----------------  G-buffer --------------*/
    if (m_GDepthRBO)
    {
        glDeleteRenderbuffers(1, &m_GDepthRBO);
        m_GDepthRBO = 0;
    }

    if (m_GAlbedoAndSpecIntensity)
    {
        glDeleteTextures(1, &m_GAlbedoAndSpecIntensity);
        m_GAlbedoAndSpecIntensity = 0;
    }

    if (m_GNormalAndTangent)
    {
        glDeleteTextures(1, &m_GNormalAndTangent);
        m_GNormalAndTangent = 0;
    }

    if (m_GPositionAndSpecPower)
    {
        glDeleteTextures(1, &m_GPositionAndSpecPower);
        m_GPositionAndSpecPower = 0;
    }

	if (m_IndirectDiffuseHalfBuffer)
	{
		glDeleteTextures(1, &m_IndirectDiffuseHalfBuffer);
		m_IndirectDiffuseHalfBuffer = 0;
	}

	if (m_IndirectDiffuseFBO)
	{
		glDeleteFramebuffers(1, &m_IndirectDiffuseFBO);
		m_IndirectDiffuseFBO = 0;
	}

    if (m_GBufferFBO)
    {
        glDeleteFramebuffers(1, &m_GBufferFBO);
        m_GBufferFBO = 0;
    }
}

bool World::IsKeyDown(int key)
{
    int k = World::GetInst().GetKey(key);
    return (k == GLFW_REPEAT || k == GLFW_PRESS);
}

int World::GetKey(int key)
{
    auto key_it = m_KeyStatusMap.find(key);
    if (key_it == m_KeyStatusMap.cend())
        return GLFW_RELEASE;
    else return key_it->second;
}