// Author: Marc Comino 2020
#define STB_IMAGE_IMPLEMENTATION
#include <glwidget.h>
#include <stb_image.h>
#include <GL/glew.h>
#include <glm/gtc/matrix_transform.hpp>
#include <fstream>
#include <iostream>
#include <memory>
#include <string>
#include <chrono>
#include <ctime>  
#include "./mesh_io.h"
#include "./triangle_mesh.h"
#include <thread>
#include <functional>

namespace {

const double kFieldOfView = 60;
const double kZNear = 0.0001;
const double kZFar = 20;

const char kNormalVertexShaderFile[] = "../shaders/normal.vert";
const char kNormalFragmentShaderFile[] = "../shaders/normal.frag";

const int kVertexAttributeIdx = 0;
const int kNormalAttributeIdx = 1;



const int N_INSTANCES = 25;
glm::vec2 translations[N_INSTANCES];


bool ReadFile(const std::string filename, std::string *shader_source) {
  std::ifstream infile(filename.c_str());

  if (!infile.is_open() || !infile.good()) {
    std::cerr << "Error " + filename + " not found." << std::endl;
    return false;
  }

  std::stringstream stream;
  stream << infile.rdbuf();
  infile.close();

  *shader_source = stream.str();
  return true;
}




bool LoadProgram(const std::string &vertex, const std::string &fragment,
                 QOpenGLShaderProgram *program) {
  std::string vertex_shader, fragment_shader;
  bool res =
      ReadFile(vertex, &vertex_shader) && ReadFile(fragment, &fragment_shader);

  if (res) {
    program->addShaderFromSourceCode(QOpenGLShader::Vertex,
                                     vertex_shader.c_str());
    program->addShaderFromSourceCode(QOpenGLShader::Fragment,
                                     fragment_shader.c_str());
    program->bindAttributeLocation("vertex", kVertexAttributeIdx);
    program->bindAttributeLocation("normal", kNormalAttributeIdx);
    program->link();
    std::cerr << "Program loaded: " + vertex + "   " + fragment << std::endl;
  }
  else {
      std::cerr << "ERROR LOADING: " + vertex + "   " + fragment << std::endl;
  }
  return res;
}

}  // namespace

GLWidget::GLWidget(QWidget *parent)
    : QGLWidget(parent),
      initialized_(false),
      width_(0.0),
      height_(0.0),
      mode_(0),
      currentLod_(0) {
  setFocusPolicy(Qt::StrongFocus);
}

GLWidget::~GLWidget() {
  if (initialized_) {

  }
}/*
void GLWidget::timer_start(unsigned int interval)
{
    std::thread([this, interval]() {
        while (true)
        {
            update();
            std::this_thread::sleep_for(std::chrono::milliseconds(interval));
        }
    }).detach();
}*/

bool GLWidget::LoadModel(const QString &filename) {
  //Intances positions
  int index = 0;
  float offset = 8.0f;
  for(int y = -5; y < 5; y += 2)
  {
      for(int x = -10; x < 10; x += 2)
      {
          if(index < N_INSTANCES)
          {
            glm::vec2 translation;
            translation.x = (float)x / 0.8f + offset;
            translation.y = (float)y / 0.8f + offset/2;
            translations[index++] = translation;
          }
      }
  } 

  std::string file = filename.toUtf8().constData();
  size_t pos = file.find_last_of(".");
  std::string type = file.substr(pos + 1);

  std::unique_ptr<data_representation::TriangleMesh> mesh = std::make_unique<data_representation::TriangleMesh>();
  

  bool res = false;
  if (type.compare("ply") == 0) {
    res = data_representation::ReadFromPly(file, mesh.get());
  }
  std::cout << "..................." << std::endl;
  if (res) {
    std::vector<data_representation::Node*> nodes;
    data_representation::Node* octreeFather = data_representation::GenOctree(mesh.get(),nodes);
    for(int l=0; l<3;++l)
    {
      for(int i= 0; i<=N_LODS; i++)
      {
        std::cout << "Iteration " <<  i << std::endl;
          std::unique_ptr<data_representation::TriangleMesh> meshLOD = std::make_unique<data_representation::TriangleMesh>();
          //data_representation::GenLodMeshQuadratics(0.25, mesh.get(),meshLOD.get());
          if(l==2)
            data_representation::GenLodMeshQuadratics((float)(i)/N_LODS, mesh.get(),meshLOD.get());
          else if(l==1)
          {
            if(i==0) data_representation::GenLodMeshAverageOctree(-1, mesh.get(),meshLOD.get(), nodes);
            else data_representation::GenLodMeshAverageOctree(N_LODS+1-i-1, mesh.get(),meshLOD.get(), nodes);
          }
          else
            data_representation::GenLodMeshAverage((float)(i)/N_LODS, mesh.get(),meshLOD.get());

          //mesh_.reset(mesh.release());
          mesh_[l*(N_LODS+1)+i].reset(meshLOD.release());
        
        
        std::cout << "Model has " <<  mesh_[i]->buffer_.size() << " elements in buffer" << std::endl;
        // TODO(students): Create / Initialize buffers.
        glGenVertexArrays(1, &modelVAO[l*(N_LODS+1)+i]);
        glGenBuffers(1, &modelVBO[l*(N_LODS+1)+i]);
        glGenBuffers(1, &modelEBO[l*(N_LODS+1)+i]);
        glBindVertexArray(modelVAO[l*(N_LODS+1)+i]);
        glBindBuffer(GL_ARRAY_BUFFER, modelVBO[l*(N_LODS+1)+i]);
        glBufferData(GL_ARRAY_BUFFER, mesh_[l*(N_LODS+1)+i]->buffer_.size()* sizeof(float), &mesh_[l*(N_LODS+1)+i]->buffer_[0], GL_STATIC_DRAW);
        glEnableVertexAttribArray(0);
        glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 6 * sizeof(float), (void*)0);
        glEnableVertexAttribArray(1);
        glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, 6 * sizeof(float), (void*)(3 * sizeof(float)));
        glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, modelEBO[l*(N_LODS+1)+i]);
        glBufferData(GL_ELEMENT_ARRAY_BUFFER, mesh_[l*(N_LODS+1)+i]->faces_.size() * sizeof(int), &mesh_[l*(N_LODS+1)+i]->faces_[0], GL_STATIC_DRAW);
    //Instancing
        //Offsets
        glGenBuffers(1, &instanceVBO);
        glBindBuffer(GL_ARRAY_BUFFER, instanceVBO);
        glBufferData(GL_ARRAY_BUFFER, sizeof(glm::vec2) * N_INSTANCES, &translations[0], GL_STATIC_DRAW);
        glBindBuffer(GL_ARRAY_BUFFER, 0); 

        glEnableVertexAttribArray(2);
        glBindBuffer(GL_ARRAY_BUFFER, instanceVBO); // this attribute comes from a different vertex buffer
        glVertexAttribPointer(2, 2, GL_FLOAT, GL_FALSE, 2 * sizeof(float), (void*)0);
        glBindBuffer(GL_ARRAY_BUFFER, 0);
        glVertexAttribDivisor(2, 1); // tell OpenGL this is an instanced vertex attribute.
        // END offsets.
      }
    }
    std::cerr << "Deleting octree " << std::endl;
    //data_representation::Node::deleteOctree(nodes);

    //emit SetFaces(QString(std::to_string(mesh_->faces_.size() / 3).c_str()));
    //emit SetVertices(
    //    QString(std::to_string(mesh_->vertices_.size() / 3).c_str()));
    std::cerr << "Model loaded " + file << std::endl;
    return true;
  }
  std::cerr << "ERROR loading model " + file << std::endl;
  return false;
}



void GLWidget::initializeGL() {
  glewExperimental=true;

  glewInit();

  glEnable(GL_NORMALIZE);
  glEnable(GL_CULL_FACE);
  glCullFace(GL_BACK);
  glEnable(GL_DEPTH_TEST);



  shader_program_ = std::make_unique<QOpenGLShaderProgram>();

  bool res =
      LoadProgram(kNormalVertexShaderFile, kNormalFragmentShaderFile,
                  shader_program_.get());
  
  if (!res) exit(0);
  std::cerr << "All programs loaded" << std::endl;

 
  initialized_ = true;
  std::cerr << "Gl init OK" << std::endl;

  glDepthFunc(GL_LEQUAL);
  // enable seamless cubemap sampling for lower mip levels in the pre-filter map.
  glEnable(GL_TEXTURE_CUBE_MAP_SEAMLESS);


  LoadModel("../models/sphere.ply");
  std::cerr << "Default model loaded OK" << std::endl;
  //timer_start(1000);
}

void GLWidget::reloadShaders()
{
    shader_program_.reset();
    shader_program_ = std::make_unique<QOpenGLShaderProgram>();
    LoadProgram(kNormalVertexShaderFile, kNormalFragmentShaderFile,
                shader_program_.get());


}

void GLWidget::resizeGL(int w, int h) {
  if (h == 0) h = 1;
  width_ = w;
  height_ = h;

  camera_.SetViewport(0, 0, w, h);
  camera_.SetProjection(kFieldOfView, kZNear, kZFar);
}

void GLWidget::mousePressEvent(QMouseEvent *event) {
  if (event->button() == Qt::LeftButton) {
    camera_.StartRotating(event->x(), event->y());
  }
  if (event->button() == Qt::RightButton) {
    camera_.StartZooming(event->x(), event->y());
  }
  update();
}

void GLWidget::mouseMoveEvent(QMouseEvent *event) {
  camera_.SetRotationX(event->y());
  camera_.SetRotationY(event->x());
  camera_.SafeZoom(event->y());
  update();
}

void GLWidget::mouseReleaseEvent(QMouseEvent *event) {
  if (event->button() == Qt::LeftButton) {
    camera_.StopRotating(event->x(), event->y());
  }
  if (event->button() == Qt::RightButton) {
    camera_.StopZooming(event->x(), event->y());
  }
  update();
}

void GLWidget::keyPressEvent(QKeyEvent *event) {
  if (event->key() == Qt::Key_Up) camera_.Zoom(-1);
  if (event->key() == Qt::Key_Down) camera_.Zoom(1);

  if (event->key() == Qt::Key_Left) camera_.Rotate(-1);
  if (event->key() == Qt::Key_Right) camera_.Rotate(1);

  if (event->key() == Qt::Key_W) camera_.Zoom(-1);
  if (event->key() == Qt::Key_S) camera_.Zoom(1);

  if (event->key() == Qt::Key_A) camera_.Rotate(-1);
  if (event->key() == Qt::Key_D) camera_.Rotate(1);

  if (event->key() == Qt::Key_R) {
    reloadShaders();
  }

  update();
}

void GLWidget::paintGL() {
  glClearColor(1.0f, 1.0f, 1.0f, 1.0f);
  glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

  if (initialized_) {
    //camera_.UpdateModel(mesh_[currentLod_]->min_, mesh_[currentLod_]->max_);
    camera_.SetViewport();

    Eigen::Matrix4f projection = camera_.SetProjection();
    Eigen::Matrix4f view = camera_.SetView();
    Eigen::Matrix4f model = camera_.SetModel();
   /* Eigen::Matrix4f t = view* model;
    Eigen::Matrix3f normal;
    for (int i = 0; i < 3; ++i)
      for (int j = 0; j < 3; ++j) normal(i, j) = t(i, j);

    normal = normal.inverse().transpose();
*/
    //std::cerr << "Painting lod " << currentLod_ << " mode " << mode_<< std::endl;

    if (mesh_[mode_*(N_LODS+1)+currentLod_] != nullptr) {
      GLint projection_location, view_location, model_location;


        shader_program_->bind();
        projection_location =
            shader_program_->uniformLocation("projection");
        view_location = shader_program_->uniformLocation("view");
        model_location = shader_program_->uniformLocation("model");

      

      glUniformMatrix4fv(projection_location, 1, GL_FALSE, projection.data());
      glUniformMatrix4fv(view_location, 1, GL_FALSE, view.data());
      glUniformMatrix4fv(model_location, 1, GL_FALSE, model.data());






      glBindVertexArray(modelVAO[mode_*(N_LODS+1)+currentLod_]);
      //glDrawElements(GL_TRIANGLES, mesh_->faces_.size(), GL_UNSIGNED_INT, 0);
      glDrawElementsInstanced(GL_TRIANGLES, mesh_[mode_*(N_LODS+1)+currentLod_]->faces_.size(), GL_UNSIGNED_INT, 0, N_INSTANCES);

      //glDrawArraysInstanced(GL_TRIANGLES, 0, mesh_->faces_.size(), N_INSTANCES); 

      glBindVertexArray(0);




    }
  }
}
void GLWidget::update()
{
  auto start = std::chrono::system_clock::now();
  updateGL();
  auto end = std::chrono::system_clock::now();
  std::chrono::duration<double> elapsed_seconds = end-start;
  //std::time_t end_time = std::chrono::system_clock::to_time_t(end);
  emit SetFramerate(QString(std::to_string(1/elapsed_seconds.count()).c_str()));
}

void GLWidget::SetAverage(bool set) {
  mode_ = 0;
  update();
}
void GLWidget::SetOctree(bool set) {
  mode_ = 1;
  update();
}
void GLWidget::SetQuadratics(bool set) {
  mode_ = 2;
  update();
}


void GLWidget::SetLod1(bool set) {
  currentLod_ = 0;
  update();
}
void GLWidget::SetLod2(bool set) {
  currentLod_ = 1;
  update();
}
void GLWidget::SetLod3(bool set) {
  currentLod_ = 2;
  update();
}
void GLWidget::SetLod4(bool set) {
  currentLod_ = 3;
  update();
}
void GLWidget::SetLod5(bool set) {
  currentLod_ = 4;
  update();
}


