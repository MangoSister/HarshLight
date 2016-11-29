#include "World.h"
#include <cassert>

World::~World()
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
}

void World::MouseCallback(GLFWwindow * window, double xpos, double ypos)
{
	World& world = World::GetInst();
	static double s_LastMouseX = 0.5 * static_cast<double>(world.m_ViewportWidth);
	static double s_LastMouseY = 0.5 * static_cast<double>(world.m_ViewportHeight);

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

void World::SetWindow(GLFWwindow* window, uint32_t width, uint32_t height)
{
    m_Window = window;
	glfwSetInputMode(window, GLFW_CURSOR, GLFW_CURSOR_DISABLED);
	glfwSetCursorPosCallback(window, MouseCallback);
	m_ViewportWidth = width;
	m_ViewportHeight = height;
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

void World::Start()
{
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
    if (m_VoxelizeTex)
    {
        m_VoxelizeTex->CleanContent();

        glDisable(GL_DEPTH_TEST);
        glDisable(GL_CULL_FACE);
		glPolygonMode(GL_FRONT_AND_BACK, GL_FILL);
        glColorMask(GL_FALSE, GL_FALSE, GL_FALSE, GL_FALSE);
        glDepthMask(GL_FALSE);
        glViewport(0, 0, 256, 256);

        if (m_VoxelizeCamera)
            m_VoxelizeCamera->UpdateCamMtx(UniformBufferBinding::kMainCam);
        else
            fprintf(stderr, "WARNING: VoxelizeCamera is null\n");

		for (ModelRenderer* renderer : m_Renderers)
		{
			renderer->Render(RenderPass::kVoxelize);
			glMemoryBarrier(GL_ALL_BARRIER_BITS);
		}

        glEnable(GL_DEPTH_TEST);
        glEnable(GL_CULL_FACE);
        glColorMask(GL_TRUE, GL_TRUE, GL_TRUE, GL_TRUE);
        glDepthMask(GL_TRUE);
        glViewport(0, 0, m_ViewportWidth, m_ViewportHeight);

		
    }
}

void World::MainLoop()
{
	m_CurrTime = std::chrono::system_clock::now();
	float elapsed = static_cast<float>((std::chrono::duration<double>(m_CurrTime - m_LastTime)).count());
	//fps: 1 / elapsed
	m_LastTime = m_CurrTime;

    /*--------- CPU update ---------*/
    for (Component* comp : m_Components)
    {
        assert(comp != nullptr);
        comp->Update(elapsed);
    }

    if (GetKey(GLFW_KEY_Z) == GLFW_PRESS)
        m_RenderPassSwitch[0] = !m_RenderPassSwitch[0];
    if (GetKey(GLFW_KEY_X) == GLFW_PRESS)
        m_RenderPassSwitch[1] = !m_RenderPassSwitch[1];

    glBindFramebuffer(GL_FRAMEBUFFER, 0);
    glClearColor(0.2f, 0.3f, 0.5f, 1.0f);
   // glClearColor(0.0f, 0.0f, 1.0f, 1.0f);
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
    glEnable(GL_DEPTH_TEST);
	glEnable(GL_CULL_FACE);
    glViewport(0, 0, m_ViewportWidth, m_ViewportHeight);
   
	//glGetTexImage(m_TexObject, 0, )

    /*--------- pass 1: regular render to default frame buffer ---------*/
    if (m_RenderPassSwitch[0])
    {
        if (m_MainCamera)
            m_MainCamera->UpdateCamMtx(UniformBufferBinding::kMainCam);
        else
            fprintf(stderr, "WARNING: MainCamera is null\n");
        
        //reconstruct voxelize space
        if (m_VoxelizeCamera)
            m_VoxelizeCamera->UpdateCamMtx(UniformBufferBinding::kVoxelSpaceReconstruct);
        else
            fprintf(stderr, "WARNING: VoxelizeCamera is null\n");

        for (ModelRenderer* renderer : m_Renderers)
            renderer->Render(RenderPass::kRegular);
    }

    if (m_RenderPassSwitch[1])
    {
        /*--------- pass 2: regular render to frame buffer displays ---------*/
        if (m_VoxelizeCamera)
            m_VoxelizeCamera->UpdateCamMtx(UniformBufferBinding::kMainCam);
        else
            fprintf(stderr, "WARNING: VoxelizeCamera is null\n");

        //reconstruct voxelize space
        if (m_VoxelizeCamera)
            m_VoxelizeCamera->UpdateCamMtx(UniformBufferBinding::kVoxelSpaceReconstruct);
        else
            fprintf(stderr, "WARNING: VoxelizeCamera is null\n");

        for (FrameBufferDisplay* display : m_FrameBufferDisplays)
        {
            assert(display != nullptr);
            display->StartRenderToFrameBuffer();
            for (ModelRenderer* renderer : m_Renderers)
                renderer->Render(RenderPass::kRegular);
        }

        /*--------- pass 3: render frame buffer displays as overlay ---------*/
        if (m_MainCamera)
            m_MainCamera->UpdateCamMtx(UniformBufferBinding::kMainCam);
        else
            fprintf(stderr, "WARNING: MainCamera is null\n");

        glBindFramebuffer(GL_FRAMEBUFFER, 0);
        glDisable(GL_DEPTH_TEST);
        glDisable(GL_CULL_FACE);
        glPolygonMode(GL_FRONT_AND_BACK, GL_FILL);
        glViewport(0, 0, m_ViewportWidth, m_ViewportHeight);
        for (FrameBufferDisplay* display : m_FrameBufferDisplays)
        {
            assert(display != nullptr);
            display->Render(RenderPass::kPost);
        }
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
        uint32_t albedo_num = scene->mMaterials[i]->GetTextureCount(aiTextureType_DIFFUSE);
//#ifdef _DEBUG
//        assert(albedo_num > 0); //temporary
//#endif
        if (albedo_num > 0)
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

int World::GetKey(int key)
{
    return glfwGetKey(m_Window, key);
}