// Author: Marc Comino 2020

#include <mesh_io.h>

#include <assert.h>

#include <algorithm>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>
#include <cmath> 
#include "./triangle_mesh.h"

#include <bits/stdc++.h>
namespace data_representation {

static const int DEBUG_INT = -1;
Node::Node(const std::vector<float> &vertices, const std::vector<int> &currentIndices, std::vector<Node*> &nodes, int depth, Node* father, double offset_x, double offset_y, double offset_z, double size_x, double size_y, double size_z)
: depth_(depth+1)
, father_(father)
, vertex_index_(-1)
{
  
  if(depth_ < DEBUG_INT) std::cout << "\t Creating Node, depth: "<< depth_ << std::endl;
  //check if we reached the top of the tree
  if(currentIndices.size()==1) {
    if(depth_ < DEBUG_INT)  std::cout << "\t we reached the top, index: "<< currentIndices[0] << std::endl;
    assert(vertex_index_==-1);
    vertex_index_ = currentIndices[0];
    nodes[currentIndices[0]] = this;
  }
  else {
    assert(currentIndices.size()>1);
    if(depth_ < DEBUG_INT)  std::cout << "\t we need to go deeper: "<< vertices.size()/3 << " "<< currentIndices.size() << std::endl;
    //we still have more than one primitives
    //                  000       001       010      011        100       101       110       111
    std::vector<int> indices1, indices2, indices3, indices4, indices5, indices6, indices7, indices8;
    for(int i =0; i<currentIndices.size(); ++i)
    {
      float x_vert = vertices[currentIndices[i]*3];
      float y_vert = vertices[currentIndices[i]*3+1];
      float z_vert = vertices[currentIndices[i]*3+2];
      int occ_x = x_vert >= offset_x +size_x/2. ? 1:0;
      int occ_y = y_vert >= offset_y +size_y/2. ? 1:0;
      int occ_z = z_vert >= offset_z +size_z/2. ? 1:0;
      if(depth_ < DEBUG_INT) std::cout << "\t\t\t offset: "<< size_x << " " << size_y <<" " << size_z << std::endl;
      if(depth_ < DEBUG_INT) std::cout << "\t\t\t offset: "<< offset_x << " " << offset_y <<" " << offset_z << std::endl;
      if(depth_ < DEBUG_INT) std::cout << "\t\t\t pos: "<< x_vert << " " << y_vert <<" " << z_vert << std::endl;
      if(depth_ < DEBUG_INT) std::cout << "\t\t\t occ pos: "<< occ_x << " " << occ_y <<" " << occ_z << std::endl;
      assert(x_vert <= offset_x +size_x);
      assert(y_vert <= offset_y +size_y);
      assert(z_vert <= offset_z +size_z);

      assert(x_vert >= offset_x);
      assert(y_vert >= offset_y);
      assert(z_vert >= offset_z);

      


      if(occ_x)
      {
        if(occ_y)
        {
          if(occ_z)
          {
            indices8.push_back(currentIndices[i]);
          }
          else
          {
            indices7.push_back(currentIndices[i]);
          }
        }
        else
        {
          if(occ_z)
          {
            indices6.push_back(currentIndices[i]);
          }
          else
          {
            indices5.push_back(currentIndices[i]);
          }
        }
      }
      else
      {
        if(occ_y)
        {
          if(occ_z)
          {
            indices4.push_back(currentIndices[i]);
          }
          else
          {
            indices3.push_back(currentIndices[i]);
          }
        }
        else
        {
          if(occ_z)
          {
            indices2.push_back(currentIndices[i]);
          }
          else
          {
            indices1.push_back(currentIndices[i]);
          }
        }
      }
    }
    assert(indices1.size() + indices2.size() +indices3.size() + indices4.size() + indices5.size() + indices6.size() + indices7.size() + indices8.size() == currentIndices.size());
    if(depth_ < DEBUG_INT)  std::cout << "\t\t node1 list size: "<< indices1.size() << std::endl;
    if(depth_ < DEBUG_INT) std::cout << "\t\t node2 list size: "<< indices2.size() << std::endl;
    if(depth_ < DEBUG_INT) std::cout << "\t\t node3 list size: "<< indices3.size() << std::endl;
    if(depth_ < DEBUG_INT) std::cout << "\t\t node4 list size: "<< indices4.size() << std::endl;
    if(depth_ < DEBUG_INT) std::cout << "\t\t node5 list size: "<< indices5.size() << std::endl;
    if(depth_ < DEBUG_INT) std::cout << "\t\t node6 list size: "<< indices6.size() << std::endl;
    if(depth_ < DEBUG_INT) std::cout << "\t\t node7 list size: "<< indices7.size() << std::endl;
    if(depth_ < DEBUG_INT) std::cout << "\t\t node8 list size: "<< indices8.size() << std::endl;
    if(indices1.size() > 0) new Node(vertices, indices1, nodes, depth_, this, offset_x+0*(size_x/2), offset_y+0*(size_y/2), offset_z+0*(size_z/2), size_x/2., size_y/2., size_z/2.);
    if(indices2.size() > 0) new Node(vertices, indices2, nodes, depth_, this, offset_x+0*(size_x/2), offset_y+0*(size_y/2), offset_z+1*(size_z/2), size_x/2., size_y/2., size_z/2.);
    if(indices3.size() > 0) new Node(vertices, indices3, nodes, depth_, this, offset_x+0*(size_x/2), offset_y+1*(size_y/2), offset_z+0*(size_z/2), size_x/2., size_y/2., size_z/2.);
    if(indices4.size() > 0) new Node(vertices, indices4, nodes, depth_, this, offset_x+0*(size_x/2), offset_y+1*(size_y/2), offset_z+1*(size_z/2), size_x/2., size_y/2., size_z/2.);
    if(indices5.size() > 0) new Node(vertices, indices5, nodes, depth_, this, offset_x+1*(size_x/2), offset_y+0*(size_y/2), offset_z+0*(size_z/2), size_x/2., size_y/2., size_z/2.);
    if(indices6.size() > 0) new Node(vertices, indices6, nodes, depth_, this, offset_x+1*(size_x/2), offset_y+0*(size_y/2), offset_z+1*(size_z/2), size_x/2., size_y/2., size_z/2.);
    if(indices7.size() > 0) new Node(vertices, indices7, nodes, depth_, this, offset_x+1*(size_x/2), offset_y+1*(size_y/2), offset_z+0*(size_z/2), size_x/2., size_y/2., size_z/2.);
    if(indices8.size() > 0) new Node(vertices, indices8, nodes, depth_, this, offset_x+1*(size_x/2), offset_y+1*(size_y/2), offset_z+1*(size_z/2), size_x/2., size_y/2., size_z/2.);
  }
}
void Node::deleteOctree(std::vector<Node*> &nodes)
{
  int depth=-1;
  for(int i=0; i< nodes.size(); ++i)
  {
    if(nodes[i]->depth_>depth) depth = nodes[i]->depth_;
  }
  std::vector<std::set<Node*> > nodesLevels(depth+1,std::set<Node*>());
  for(int i=0; i< nodes.size(); ++i)
  {
    std::cout << "\t\t "<< nodes[i]->depth_ << std::endl;
    nodesLevels[nodes[i]->depth_].insert(nodes[i]);
  }
  for(int i= nodesLevels.size()-1; i>=0; ++i)
  {
    std::set<Node*>::iterator it;
    for (it=nodesLevels[i].begin(); it!=nodesLevels[i].end(); ++it)
    {
      Node* currNode = *it;
      if(currNode != nullptr)
      {
        if(currNode->father_ !=nullptr)
        {
          Node* fatherNode = currNode->father_;
          nodesLevels[fatherNode->depth_].insert(fatherNode);
        }
        nodesLevels[i].erase(it);
        delete(currNode);
        it--;
      }
    }
  }
}

Node* GenOctree(const TriangleMesh *mesh, std::vector<Node*> &nodes)
{
  Node* n = nullptr;
  std::vector<int> indices;
  for(int i=0; i<mesh->vertices_.size(); i = i +3)
  {
    indices.push_back(i/3);
    nodes.push_back(nullptr);
    //std::cout << mesh->vertices_[i] << " " << mesh->vertices_[i+1]<< " " << mesh->vertices_[i+2]<< std::endl;
  }
  n = new Node(mesh->vertices_,indices,nodes,-1,nullptr,mesh->min_[0],mesh->min_[1],mesh->min_[2],mesh->max_[0]-mesh->min_[0],mesh->max_[1]-mesh->min_[1],mesh->max_[2]-mesh->min_[2]);
  return n;
}
namespace {




//Structures for quadratics method
class face;
class vertex {
public:
  double x, y, z;
  double **Q;
  std::set<int> face_list;
  vertex(double _x = 0., double _y = 0., double _z = 0.): x(_x), y(_y), z(_z) {
    Q = new double*[4];
    for (int i = 0; i < 4; i++) {
      Q[i] = new double[4];
      Q[i][0] = Q[i][1] = Q[i][2] = Q[i][3] = 0;
    }
  }
  void computeQ(const std::vector<face> &faces);
};

class face {
public:
  int v1, v2, v3;
  double a, b, c, d;
  face(int _v1 = 0, int _v2 = 0, int _v3 = 0): v1(_v1), v2(_v2), v3(_v3) {}
  void computePara(const std::vector<vertex> &vertices);
};




void vertex::computeQ(const std::vector<face> &faces) {


  for (int i = 0; i < 4; i++)
    {
      for (int j = 0; j < 4; j++) Q[i][j] = 0;
    }

  for (auto it = face_list.begin(); it != face_list.end(); it++) {
    double *p = new double[4];
    p[0] = faces[*it].a; p[1] = faces[*it].b; p[2] = faces[*it].c; p[3] = faces[*it].d;

    for (int i = 0; i < 4; i++)
      for (int j = 0; j < 4; j++) 
        {
          Q[i][j] += p[i] * p[j];
        }
    delete [] p;
  }
}

void face::computePara(const std::vector<vertex> &vertices) {
  double x1 = vertices[v2].x - vertices[v1].x, y1 = vertices[v2].y - vertices[v1].y, z1 = vertices[v2].z - vertices[v1].z;
  double x2 = vertices[v3].x - vertices[v1].x, y2 = vertices[v3].y - vertices[v1].y, z2 = vertices[v3].z - vertices[v1].z;
  a = y1 * z2 - y2 * z1; b = z1 * x2 - z2 * x1; c = x1 * y2 - x2 * y1;
  double len = sqrt(a * a + b * b + c * c);
  a /= len; b /= len; c /= len;
  d = -(a * vertices[v1].x + b * vertices[v1].y + c * vertices[v1].z);
}




double computeNewPosition(double *_x, double *_y, double *_z,const std::vector<vertex> &vertices, const std::vector<int> &indices) {

  double **A = new double*[4];
  for (int i = 0; i < 4; i++) {
    A[i] = new double[4];
    for (int j = 0; j < 4; j++) A[i][j] = 0;
  }
  for(int in=0; in<indices.size(); ++in)
  {
    for (int i = 0; i < 4; i++) {
      A[i] = new double[4];
      for (int j = 0; j < 4; j++) A[i][j] += vertices[indices[in]].Q[i][j];
    }
  }
  A[3][0] = A[3][1] = A[3][2] = 0; A[3][3] = 1;
  double *b = new double[4];
  b[0] = b[1] = b[2] = 0; b[3] = 1;
  double *x = new double[4];
  x[0] = x[1] = x[2] = x[3] = 0;
  bool flag = true;
  for (int i = 0; i < 4; i++) {
    int maxLine = i;
    for (int j = i + 1; j < 4; j++)
      if (fabs(A[j][i]) > fabs(A[maxLine][i])) maxLine = j;
    for (int j = i; j < 4; j++) std::swap(A[i][j], A[maxLine][j]);
    std::swap(b[i], b[maxLine]);
    double t = A[i][i];
    if (fabs(t) < 1e-10) {
      for(int in=0; in<indices.size(); ++in)
      {
        x[0] += vertices[indices[in]].x;
        x[1] += vertices[indices[in]].y;
        x[2] += vertices[indices[in]].z;
      }
      x[0] = x[0] / indices.size();
      x[1] = x[1] / indices.size();
      x[2] = x[2] / indices.size();
      x[3] = 1.;
      flag = false;
      break;
    }
    for (int j = i; j < 4; j++) A[i][j] /= t;
    b[i] /= t;
    for (int j = i + 1; j < 4; j++) if (fabs(A[j][i]) > 1e-8) {
      t = A[j][i];
      for (int k = i; k < 4; k++) A[j][k] -= A[i][k] * t;
      b[j] -= b[i] * t;
    }
  }
  if (flag) {
    for (int i = 3; i >= 0; i--) {
      x[i] = b[i];
      for (int k = i + 1; k < 4; k++) x[i] -= A[i][k] * x[k];
    }
  }
  assert(fabs(x[3] - 1.) < 1e-8);
  *_x = x[0]; *_y = x[1]; *_z = x[2];
  double cost = 0;
  for (int i = 0; i < 4; i++) {
    for (int j = 0; j < 4; j++)
    {
      double sum_aux =0;
      for(int in=0; in<indices.size(); ++in)
      {
        sum_aux += vertices[indices[in]].Q[i][j];
      }
      cost += x[i] * x[j] * sum_aux;
    }
    delete [] A[i];
  }
  delete [] A;
  delete [] x;
  delete [] b;
  return cost;
}





////////////////////////////////

template <typename T>
void Add3Items(T i1, T i2, T i3, size_t index, std::vector<T> *vector) {
  (*vector)[index] = i1;
  (*vector)[index + 1] = i2;
  (*vector)[index + 2] = i3;
}

bool ReadPlyHeader(std::ifstream *fin, int *vertices, int *faces) {
  char line[100];

  fin->getline(line, 100);
  if (strncmp(line, "ply", 3) != 0) return false;

  *vertices = 0;
  fin->getline(line, 100);
  while (strncmp(line, "end_header", 10) != 0) {
    if (strncmp(line, "element vertex", 14) == 0) *vertices = atoi(&line[15]);
    fin->getline(line, 100);
    if (strncmp(line, "element face", 12) == 0) *faces = atoi(&line[13]);
  }

  if (*vertices <= 0) return false;

  std::cout << "Loading triangle mesh" << std::endl;
  std::cout << "\tVertices = " << *vertices << std::endl;
  std::cout << "\tFaces = " << *faces << std::endl;

  return true;
}

void ReadPlyVertices(std::ifstream *fin, TriangleMesh *mesh) {
  const size_t kVertices = mesh->vertices_.size() / 3;
  for (size_t i = 0; i < kVertices; ++i) {
    float x, y, z;
    fin->read(reinterpret_cast<char *>(&x), sizeof(float));
    fin->read(reinterpret_cast<char *>(&y), sizeof(float));
    fin->read(reinterpret_cast<char *>(&z), sizeof(float));

    Add3Items(x, y, z, i * 3, &(mesh->vertices_));
  }
}

void ReadPlyFaces(std::ifstream *fin, TriangleMesh *mesh) {
  unsigned char vertex_per_face;

  const size_t kFaces = mesh->faces_.size() / 3;
  for (size_t i = 0; i < kFaces; ++i) {
    int v1, v2, v3;
    fin->read(reinterpret_cast<char *>(&vertex_per_face),
              sizeof(unsigned char));
    assert(vertex_per_face == 3);

    fin->read(reinterpret_cast<char *>(&v1), sizeof(int));
    fin->read(reinterpret_cast<char *>(&v2), sizeof(int));
    fin->read(reinterpret_cast<char *>(&v3), sizeof(int));
    Add3Items(v1, v2, v3, i * 3, &(mesh->faces_));
  }
}

void ComputeVertexNormals(const std::vector<float> &vertices, const std::vector<int> &faces, std::vector<float> *normals) {
  std::cout << "\t Computing Vertex Normals" << std::endl;
  const size_t kFaces = faces.size();
  std::vector<float> face_normals(kFaces, 0);

  for (size_t i = 0; i < kFaces; i += 3) {
    Eigen::Vector3d v1(vertices[faces[i] * 3], vertices[faces[i] * 3 + 1],
                       vertices[faces[i] * 3 + 2]);
    Eigen::Vector3d v2(vertices[faces[i + 1] * 3],
                       vertices[faces[i + 1] * 3 + 1],
                       vertices[faces[i + 1] * 3 + 2]);
    Eigen::Vector3d v3(vertices[faces[i + 2] * 3],
                       vertices[faces[i + 2] * 3 + 1],
                       vertices[faces[i + 2] * 3 + 2]);
    Eigen::Vector3d v1v2 = v2 - v1;
    Eigen::Vector3d v1v3 = v3 - v1;
    Eigen::Vector3d normal = v1v2.cross(v1v3);

    if (normal.norm() < 0.00001) {
      normal = Eigen::Vector3d(0.0, 0.0, 0.0);
    } else {
      normal.normalize();
    }

    for (size_t j = 0; j < 3; ++j) face_normals[i + j] = normal[j];
  }
  const size_t kVertices = vertices.size();
  normals->resize(kVertices, 0);
  for (size_t i = 0; i < kFaces; i += 3) {
    for (size_t j = 0; j < 3; ++j) {
      size_t idx = static_cast<size_t>(faces[i + j]);
      Eigen::Vector3d v1(vertices[faces[i + j] * 3],
                         vertices[faces[i + j] * 3 + 1],
                         vertices[faces[i + j] * 3 + 2]);
      Eigen::Vector3d v2(vertices[faces[i + (j + 1) % 3] * 3],
                         vertices[faces[i + (j + 1) % 3] * 3 + 1],
                         vertices[faces[i + (j + 1) % 3] * 3 + 2]);
      Eigen::Vector3d v3(vertices[faces[i + (j + 2) % 3] * 3],
                         vertices[faces[i + (j + 2) % 3] * 3 + 1],
                         vertices[faces[i + (j + 2) % 3] * 3 + 2]);

      Eigen::Vector3d v1v2 = v2 - v1;
      Eigen::Vector3d v1v3 = v3 - v1;
      double angle = acos(v1v2.dot(v1v3) / (v1v2.norm() * v1v3.norm()));

      if (angle == angle) {
        for (size_t k = 0; k < 3; ++k) {
          (*normals)[idx * 3 + k] += face_normals[i + k] * angle;
        }
      }
    }
  }
  const size_t kNormals = normals->size();
  for (size_t i = 0; i < kNormals; i += 3) {
    Eigen::Vector3d normal((*normals)[i], (*normals)[i + 1], (*normals)[i + 2]);
    if (normal.norm() > 0) {
      normal.normalize();
    } else {
      normal = Eigen::Vector3d(0, 0, 0);
    }

    for (size_t j = 0; j < 3; ++j) (*normals)[i + j] = normal[j];
  }
}

void ComputeBoundingBox(const std::vector<float> vertices, TriangleMesh *mesh) {
  const size_t kVertices = vertices.size() / 3;
  for (size_t i = 0; i < kVertices; ++i) {
    mesh->min_[0] = std::min(mesh->min_[0], vertices[i * 3]);
    mesh->min_[1] = std::min(mesh->min_[1], vertices[i * 3 + 1]);
    mesh->min_[2] = std::min(mesh->min_[2], vertices[i * 3 + 2]);

    mesh->max_[0] = std::max(mesh->max_[0], vertices[i * 3]);
    mesh->max_[1] = std::max(mesh->max_[1], vertices[i * 3 + 1]);
    mesh->max_[2] = std::max(mesh->max_[2], vertices[i * 3 + 2]);


  }
}

}  // namespace

bool ReadFromPly(const std::string &filename, TriangleMesh *mesh) {
  std::ifstream fin;

  fin.open(filename.c_str(), std::ios_base::in | std::ios_base::binary);
  if (!fin.is_open() || !fin.good()) return false;

  int vertices = 0, faces = 0;
  if (!ReadPlyHeader(&fin, &vertices, &faces)) {
    fin.close();
    std::cout << "\tError loading headers " << std::endl;
    return false;
  }
  std::cout << "\tHeaders loaded " << std::endl;
  mesh->vertices_.resize(static_cast<size_t>(vertices) * 3);
  ReadPlyVertices(&fin, mesh);
  std::cout << "\tLoaded vertices " << std::endl;
  mesh->faces_.resize(static_cast<size_t>(faces) * 3);
  ReadPlyFaces(&fin, mesh);
  std::cout << "\tLoaded faces " << std::endl;
  fin.close();

  ComputeVertexNormals(mesh->vertices_, mesh->faces_, &mesh->normals_);
  std::cout << "\tGenerated normals " << std::endl;
  ComputeBoundingBox(mesh->vertices_, mesh);
  std::cout << "\tGenerated bounding box " << std::endl;
  std::cout << "\t" << mesh->max_[0] - mesh->min_[0] <<std::endl;
  std::cout << "\t" << mesh->max_[1] - mesh->min_[1] <<std::endl;
  std::cout << "\t" << mesh->max_[2] - mesh->min_[2] <<std::endl;
  mesh->prepareVertexBuffer();
  std::cout << "\tPrepared vertex buffer " << std::endl;
  return true;
}




void GenLodMeshQuadratics(float cellSize, const TriangleMesh *mesh, TriangleMesh *newMesh)
{
  newMesh->Clear();
  if(cellSize==0)
  {
    newMesh->faces_ = mesh->faces_;
    newMesh->vertices_ = mesh->vertices_;
    newMesh->normals_ = mesh->normals_;
    newMesh->buffer_ = mesh->buffer_;
    newMesh->min_ = mesh->min_;
    newMesh->max_ = mesh->max_;


      ComputeVertexNormals(newMesh->vertices_, newMesh->faces_, &newMesh->normals_);
      std::cout << "\tGenerated normals " << std::endl;
      ComputeBoundingBox(newMesh->vertices_, newMesh);
      std::cout << "\tGenerated bounding box " << std::endl;
      newMesh->prepareVertexBuffer();
      std::cout << "\tPrepared vertex buffer " << std::endl;
    return;
  }
  for(unsigned int i = 0; i < mesh->faces_.size(); ++i)
  {
    newMesh->faces_.push_back(mesh->faces_[i]);
  }

  std::cerr << "\tGenerating Lod using quadratics error: " << std::endl;
  int m_face_num = mesh->faces_.size()/3;
  int m_vertex_num = mesh->vertices_.size()/3;
  std::vector<face> m_faces;
  std::vector<vertex> m_vertices;
  //double m_simp_rate, m_dist_eps;

  ///// CREATE GRID /////
  float sizeBoundBox_x = mesh->max_[0] - mesh->min_[0];
  float sizeBoundBox_y = mesh->max_[1] - mesh->min_[1];
  float sizeBoundBox_z = mesh->max_[2] - mesh->min_[2];
  int nCellsX = ceil(sizeBoundBox_x/cellSize);
  int nCellsY = ceil(sizeBoundBox_y/cellSize);
  int nCellsZ = ceil(sizeBoundBox_z/cellSize);
  std::cout << "\t\t NumCells X Y Z: "<< nCellsX << " "  << nCellsY<< " " << nCellsZ << std::endl;
  if(nCellsX < 1 || nCellsY < 1 || nCellsZ < 1 )
  {
    std::cerr << "\tGrid too small to generate LOD" << std::endl;
    exit(0);
  }
  std::vector<std::vector<std::vector<std::vector<int> > > > cells(nCellsX, std::vector<std::vector<std::vector<int> > >(nCellsY,std::vector<std::vector<int> >(nCellsZ,std::vector<int>(0))));
  std::vector<std::vector<std::vector<std::vector<int> > > > cellsExtraFaces(nCellsX, std::vector<std::vector<std::vector<int> > >(nCellsY,std::vector<std::vector<int> >(nCellsZ,std::vector<int>(0))));
  //////////////////////
  std::cerr << "\t\t Generating auxiliar structures: " << std::endl;
  ////// GENERATE VERT AUXILIAR STRUCTURE //////
  // Vertices
  for(unsigned int i =0; i < mesh->vertices_.size(); i = i+3)
  {
    m_vertices.push_back(vertex(mesh->vertices_[i], mesh->vertices_[i+1], mesh->vertices_[i+2]));
  }

  std::cerr << "\t\t Vertices generated " << std::endl;
  //Faces
  for(unsigned int i =0; i < mesh->faces_.size(); i = i+3)
  {
    int u = mesh->faces_[i];
    int v = mesh->faces_[i+1];
    int w = mesh->faces_[i+2];
    m_vertices[u].face_list.insert(i/3);
    m_vertices[v].face_list.insert(i/3);
    m_vertices[w].face_list.insert(i/3);
    m_faces.push_back(face(u,v,w));
  }
  std::cerr << "\t\t Faces generated: " << std::endl;
  //We compute now the parametric equation of each face and the Q for each vertex
  for (int i = 0; i < m_face_num; i++) m_faces[i].computePara(m_vertices);
  for (int i = 0; i < m_vertex_num; i++) m_vertices[i].computeQ(m_faces);
  //Get the edges that are in the same cell of the uniform grid
  
  std::vector<bool> pushed_vertices(mesh->vertices_.size()/3,false);
  std::cout << "\t Space allocated, numVertices = " << pushed_vertices.size()<< std::endl;
  for(unsigned int i =0; i < mesh->faces_.size(); i = i +3)
  {
    //std::cout << "\t\t ." << std::endl;
    for(int j = 0; j < 3; ++j)
    {
      
      int indexVertex = mesh->faces_[i+j];
      //std::cout << "\t\t\t ." << indexVertex << std::endl;
      if(!pushed_vertices[indexVertex])
      {
        int cellX = std::min((int)floor((mesh->vertices_[indexVertex*3] - mesh->min_[0])/cellSize),nCellsX-1);
        int cellY = std::min((int)floor((mesh->vertices_[indexVertex*3+1] - mesh->min_[1])/cellSize),nCellsY-1);
        int cellZ = std::min((int)floor((mesh->vertices_[indexVertex*3+2] - mesh->min_[2])/cellSize),nCellsZ-1);
        //std::cout << "\t\t\t Pos " << mesh->vertices_[indexVertex] << " " << mesh->vertices_[indexVertex+1] << " " << mesh->vertices_[indexVertex+2] << std::endl;
        //std::cout << "\t\t\t Pushing Cell " << cellX << " " << cellY << " " << cellZ << std::endl;
        cells[cellX][cellY][cellZ].push_back(i+j);
        //std::cout << "\t\t\t Pushed Cell " << cellX << " " << cellY << " " << cellZ << std::endl;
        pushed_vertices[indexVertex] = true;
      }
      else 
      {
        int cellX = std::min((int)floor((mesh->vertices_[indexVertex*3] - mesh->min_[0])/cellSize),nCellsX-1);
        int cellY = std::min((int)floor((mesh->vertices_[indexVertex*3+1] - mesh->min_[1])/cellSize),nCellsY-1);
        int cellZ = std::min((int)floor((mesh->vertices_[indexVertex*3+2] - mesh->min_[2])/cellSize),nCellsZ-1);
        cellsExtraFaces[cellX][cellY][cellZ].push_back(i+j);
      }
    }
  }
  bool trobat = false;
  for(unsigned int i=0; i < pushed_vertices.size(); ++i)
  {
    if(!pushed_vertices[i]) trobat = true;
  }
  if(trobat){
    std::cerr << "\tERROR" << std::endl;
  }
  std::cout << "\t Grid done" << std::endl;


  std::cerr << "\t\t Lets do the main job: " << std::endl;
  std::cerr << "\t\t\t Vertices: " << m_vertices.size() << std::endl;
  std::cerr << "\t\t\t Faces: " << m_faces.size() << std::endl;
  





  int newIndexVertex = 0;
  newMesh->vertices_.resize(static_cast<size_t>(nCellsX*nCellsY*nCellsZ) * 3);
  for(int i =0; i < nCellsX; ++i)
  {
    for(int j =0; j < nCellsY; ++j)
    {
      for(int k =0; k < nCellsZ; ++k)
      {

        double newVertex_x =0;
        double newVertex_y =0;
        double newVertex_z =0;
        int nVerticesInCell = cells[i][j][k].size();
        std::vector<int> indicesCell(nVerticesInCell);
        for(int v = 0; v < nVerticesInCell; ++v)
        {
          int faceVertexIndex = cells[i][j][k][v];
          int vertexIndex = newMesh->faces_[faceVertexIndex];
          indicesCell[v] = newMesh->faces_[faceVertexIndex];
          newMesh->faces_[faceVertexIndex] = newIndexVertex;
        }
        

        nVerticesInCell = cellsExtraFaces[i][j][k].size();
        for(int v = 0; v < nVerticesInCell; ++v)
        {
          int faceVertexIndex = cellsExtraFaces[i][j][k][v];
          //int vertexIndex = newMesh->faces_[faceVertexIndex];

          newMesh->faces_[faceVertexIndex] = newIndexVertex;
          //std::cout << "\t Replacing " << std::endl;
        }
        //std::cerr << "\t\t Cell: " << i << " " << j << " " << k << " todo"<<  std::endl;
        double err = computeNewPosition(&newVertex_x, &newVertex_y, &newVertex_z, m_vertices, indicesCell);
        //std::cerr << "\t\t Cell: " << i << " " << j << " " << k << " done"<< std::endl;
        newMesh->vertices_[newIndexVertex*3] = newVertex_x;
        newMesh->vertices_[newIndexVertex*3+1] =newVertex_y;
        newMesh->vertices_[newIndexVertex*3+2] = newVertex_z;
        if(cells[i][j][k].size()>0)newIndexVertex +=1;
      }
    }
  }
  std::cout << "\t Collapse done" << std::endl;
  for(auto it = newMesh->faces_.begin(); it != newMesh->faces_.end(); )
  {
    if(*(it) == *(it+1) || *(it) == *(it+2) || *(it+1) == *(it+2))
    {
      it = newMesh->faces_.erase(it,it+3);
    } 
    else 
    {
      it = it+3;
    }
  }


  std::cout << "\t totalVertices " << newIndexVertex << " " << newMesh->vertices_.size() << " " << mesh->vertices_.size()/3<< std::endl;
  ComputeVertexNormals(newMesh->vertices_, newMesh->faces_, &newMesh->normals_);
  std::cout << "\tGenerated normals " << std::endl;
  ComputeBoundingBox(newMesh->vertices_, newMesh);
  std::cout << "\tGenerated bounding box " << std::endl;
  newMesh->prepareVertexBuffer();
  std::cout << "\tPrepared vertex buffer " << std::endl;

}

void GenLodMeshAverage(float cellSize, const TriangleMesh *mesh, TriangleMesh *newMesh)
{
  
  newMesh->Clear();
  if(cellSize==0)
  {
    newMesh->faces_ = mesh->faces_;
    newMesh->vertices_ = mesh->vertices_;
    newMesh->normals_ = mesh->normals_;
    newMesh->buffer_ = mesh->buffer_;
    newMesh->min_ = mesh->min_;
    newMesh->max_ = mesh->max_;


      ComputeVertexNormals(newMesh->vertices_, newMesh->faces_, &newMesh->normals_);
      std::cout << "\tGenerated normals " << std::endl;
      ComputeBoundingBox(newMesh->vertices_, newMesh);
      std::cout << "\tGenerated bounding box " << std::endl;
      newMesh->prepareVertexBuffer();
      std::cout << "\tPrepared vertex buffer " << std::endl;
    return;
  }
  std::cout << "Generating LOD with cell size: "<< cellSize<< std::endl;
  //Copy the mesh
  
  for(unsigned int i = 0; i < mesh->faces_.size(); ++i)
  {
    newMesh->faces_.push_back(mesh->faces_[i]);
  }
  ///////

  float sizeBoundBox_x = mesh->max_[0] - mesh->min_[0];
  float sizeBoundBox_y = mesh->max_[1] - mesh->min_[1];
  float sizeBoundBox_z = mesh->max_[2] - mesh->min_[2];
  int nCellsX = ceil(sizeBoundBox_x/cellSize);
  int nCellsY = ceil(sizeBoundBox_y/cellSize);
  int nCellsZ = ceil(sizeBoundBox_z/cellSize);
  std::cout << "\t NumCells X Y Z: "<< nCellsX << " "  << nCellsY<< " " << nCellsZ << std::endl;
  if(nCellsX < 1 || nCellsY < 1 || nCellsZ < 1 )
  {
    std::cerr << "\tGrid too small to generate LOD" << std::endl;
    exit(0);
  }
  //XYZ mat with vertices index
  std::vector<std::vector<std::vector<std::vector<int> > > > cells(nCellsX, std::vector<std::vector<std::vector<int> > >(nCellsY,std::vector<std::vector<int> >(nCellsZ,std::vector<int>(0))));
  std::vector<std::vector<std::vector<std::vector<int> > > > cellsExtraFaces(nCellsX, std::vector<std::vector<std::vector<int> > >(nCellsY,std::vector<std::vector<int> >(nCellsZ,std::vector<int>(0))));
  std::vector<bool> pushed_vertices(mesh->vertices_.size()/3,false);
  std::cout << "\t Space allocated, numVertices = " << pushed_vertices.size()<< std::endl;
  for(unsigned int i =0; i < mesh->faces_.size(); i = i +3)
  {
    //std::cout << "\t\t ." << std::endl;
    for(int j = 0; j < 3; ++j)
    {
      
      int indexVertex = mesh->faces_[i+j];
      //std::cout << "\t\t\t ." << indexVertex << std::endl;
      if(!pushed_vertices[indexVertex])
      {
        int cellX = std::min((int)floor((mesh->vertices_[indexVertex*3] - mesh->min_[0])/cellSize),nCellsX-1);
        int cellY = std::min((int)floor((mesh->vertices_[indexVertex*3+1] - mesh->min_[1])/cellSize),nCellsY-1);
        int cellZ = std::min((int)floor((mesh->vertices_[indexVertex*3+2] - mesh->min_[2])/cellSize),nCellsZ-1);
        //std::cout << "\t\t\t Pos " << mesh->vertices_[indexVertex] << " " << mesh->vertices_[indexVertex+1] << " " << mesh->vertices_[indexVertex+2] << std::endl;
        //std::cout << "\t\t\t Pushing Cell " << cellX << " " << cellY << " " << cellZ << std::endl;
        cells[cellX][cellY][cellZ].push_back(i+j);
        //std::cout << "\t\t\t Pushed Cell " << cellX << " " << cellY << " " << cellZ << std::endl;
        pushed_vertices[indexVertex] = true;
      }
      else 
      {
        int cellX = std::min((int)floor((mesh->vertices_[indexVertex*3] - mesh->min_[0])/cellSize),nCellsX-1);
        int cellY = std::min((int)floor((mesh->vertices_[indexVertex*3+1] - mesh->min_[1])/cellSize),nCellsY-1);
        int cellZ = std::min((int)floor((mesh->vertices_[indexVertex*3+2] - mesh->min_[2])/cellSize),nCellsZ-1);
        cellsExtraFaces[cellX][cellY][cellZ].push_back(i+j);
      }
    }
  }
  bool trobat = false;
  for(unsigned int i=0; i < pushed_vertices.size(); ++i)
  {
    if(!pushed_vertices[i]) trobat = true;
  }
  if(trobat){
    std::cerr << "\tERROR" << std::endl;
  }
  std::cout << "\t Grid done" << std::endl;
  int newIndexVertex = 0;
  newMesh->vertices_.resize(static_cast<size_t>(nCellsX*nCellsY*nCellsZ) * 3);
  for(int i =0; i < nCellsX; ++i)
  {
    for(int j =0; j < nCellsY; ++j)
    {
      for(int k =0; k < nCellsZ; ++k)
      {
        float newVertex_x =0;
        float newVertex_y =0;
        float newVertex_z =0;
        int nVerticesInCell = cells[i][j][k].size();
        for(int v = 0; v < nVerticesInCell; ++v)
        {
          int faceVertexIndex = cells[i][j][k][v];
          int vertexIndex = newMesh->faces_[faceVertexIndex];

          newMesh->faces_[faceVertexIndex] = newIndexVertex;
          //std::cout << "\t Replacing " << std::endl;
          newVertex_x += mesh->vertices_[vertexIndex*3];
          newVertex_y += mesh->vertices_[vertexIndex*3+1];
          newVertex_z += mesh->vertices_[vertexIndex*3+2];
        }
        newVertex_x = newVertex_x/nVerticesInCell;
        newVertex_y = newVertex_y/nVerticesInCell;
        newVertex_z = newVertex_z/nVerticesInCell;
        nVerticesInCell = cellsExtraFaces[i][j][k].size();
        for(int v = 0; v < nVerticesInCell; ++v)
        {
          int faceVertexIndex = cellsExtraFaces[i][j][k][v];
          //int vertexIndex = newMesh->faces_[faceVertexIndex];

          newMesh->faces_[faceVertexIndex] = newIndexVertex;
          //std::cout << "\t Replacing " << std::endl;
        }


        newMesh->vertices_[newIndexVertex*3] = newVertex_x;
        newMesh->vertices_[newIndexVertex*3+1] =newVertex_y;
        newMesh->vertices_[newIndexVertex*3+2] = newVertex_z;
        if(cells[i][j][k].size()>0)newIndexVertex +=1;
      }
    }
  }
  std::cout << "\t Collapse done" << std::endl;
  //Remove extra faces
  for(auto it = newMesh->faces_.begin(); it != newMesh->faces_.end(); )
  {
    if(*(it) == *(it+1) || *(it) == *(it+2) || *(it+1) == *(it+2))
    {
      it = newMesh->faces_.erase(it,it+3);
    } 
    else 
    {
      it = it+3;
    }
  }/*
  std::cout << "\t extra faces removed" << std::endl;
  int c = 0;
  for(int i=0; i < newMesh->faces_.size(); ++i)
  {
    std::cout << "\t" << newMesh->faces_[i] << " ";

    c++;
    if(c ==3) {
      std::cout << std::endl;
      c = 0;
    }
  }*/
  std::cout << "\t totalVertices " << newIndexVertex << " " << newMesh->vertices_.size() << " " << mesh->vertices_.size()/3<< std::endl;
  ComputeVertexNormals(newMesh->vertices_, newMesh->faces_, &newMesh->normals_);
  std::cout << "\tGenerated normals " << std::endl;
  ComputeBoundingBox(newMesh->vertices_, newMesh);
  std::cout << "\tGenerated bounding box " << std::endl;
  newMesh->prepareVertexBuffer();
  std::cout << "\tPrepared vertex buffer " << std::endl;


}


void GenLodMeshAverageOctree(int maxDepth, const TriangleMesh *mesh, TriangleMesh *newMesh, const std::vector<Node*> &octree)
{
  
  newMesh->Clear();
  if(maxDepth < 0)
  {
    newMesh->faces_ = mesh->faces_;
    newMesh->vertices_ = mesh->vertices_;
    newMesh->normals_ = mesh->normals_;
    newMesh->buffer_ = mesh->buffer_;
    newMesh->min_ = mesh->min_;
    newMesh->max_ = mesh->max_;


      ComputeVertexNormals(newMesh->vertices_, newMesh->faces_, &newMesh->normals_);
      std::cout << "\tGenerated normals " << std::endl;
      ComputeBoundingBox(newMesh->vertices_, newMesh);
      std::cout << "\tGenerated bounding box " << std::endl;
      newMesh->prepareVertexBuffer();
      std::cout << "\tPrepared vertex buffer " << std::endl;
    return;
  }

  //Copy the mesh
  for(unsigned int i = 0; i < mesh->faces_.size(); ++i)
  {
    newMesh->faces_.push_back(mesh->faces_[i]);
  }
  ///////

  std::cout << "Generating LOD with octree depth: "<< maxDepth<< std::endl;

  std::map< Node*, std::vector<int> > clusters;

  assert(octree.size() == mesh->vertices_.size()/3);

  std::vector<Node*> newOctree = octree;
  for(int i=0; i < newOctree.size(); ++i)
  {
    Node *n = newOctree[i];
    while(n->depth_ > maxDepth)
    {
      n = n->father_;
    }
    std::map<Node*, std::vector<int>>::const_iterator it = clusters.find(n);
    if(it!=clusters.end())
    {
      clusters[n].push_back(i);
      //it->second.push_back(i);
    }
    else 
    {
      std::vector<int> vec{i};
      clusters.insert( std::pair<Node*,std::vector<int> >(n,vec));
    }
  }

  //for (std::map<Node*, std::vector<int> >::iterator it=clusters.begin(); it!=clusters.end(); ++it)
  {
    //assert(it->second.size() > 1);
  }
  //  std::cout << it->first << " => " << it->second.size() << '\n';


  std::vector<std::vector<int > > facesOfVertexs(mesh->vertices_.size()/3, std::vector<int >(0));
  for(int i = 0; i < mesh->faces_.size(); i = i+3)
  {
    int v1 =  mesh->faces_[i];
    int v2 =  mesh->faces_[i+1];
    int v3 =  mesh->faces_[i+2];
    facesOfVertexs[v1].push_back(i);
    facesOfVertexs[v2].push_back(i+1);
    facesOfVertexs[v3].push_back(i+2);
  }

  newMesh->vertices_.resize(static_cast<size_t>(clusters.size()) * 3);
  //Lets do the clustering
  int newIndexVertex = 0;
  for (std::map<Node*, std::vector<int> >::iterator it=clusters.begin(); it!=clusters.end(); ++it)
  {
    float newVertex_x =0;
    float newVertex_y =0;
    float newVertex_z =0;
    int nVerticesInNode = it->second.size();
    for(int i=0; i< nVerticesInNode; ++i)
    {
      int vertexIndex = it->second[i];
      newVertex_x += mesh->vertices_[vertexIndex*3];
      newVertex_y += mesh->vertices_[vertexIndex*3+1];
      newVertex_z += mesh->vertices_[vertexIndex*3+2];
      for(int j=0; j< facesOfVertexs[vertexIndex].size(); ++j)
      {
        newMesh->faces_[facesOfVertexs[vertexIndex][j]] = newIndexVertex;
      }
    }
    newVertex_x = newVertex_x/nVerticesInNode;
    newVertex_y = newVertex_y/nVerticesInNode;
    newVertex_z = newVertex_z/nVerticesInNode;
    newMesh->vertices_[newIndexVertex*3] = newVertex_x;
    newMesh->vertices_[newIndexVertex*3+1] =newVertex_y;
    newMesh->vertices_[newIndexVertex*3+2] = newVertex_z;

    if(nVerticesInNode>0)newIndexVertex +=1;

  }
  for(auto it = newMesh->faces_.begin(); it != newMesh->faces_.end(); )
  {
    if(*(it) == *(it+1) || *(it) == *(it+2) || *(it+1) == *(it+2))
    {
      it = newMesh->faces_.erase(it,it+3);
    } 
    else 
    {
      it = it+3;
    }
  }

  std::cout << "\t totalVertices " << newIndexVertex << " " << newMesh->vertices_.size() << " " << mesh->vertices_.size()/3<< std::endl;
  ComputeVertexNormals(newMesh->vertices_, newMesh->faces_, &newMesh->normals_);
  std::cout << "\tGenerated normals " << std::endl;
  ComputeBoundingBox(newMesh->vertices_, newMesh);
  std::cout << "\tGenerated bounding box " << std::endl;
  newMesh->prepareVertexBuffer();
  std::cout << "\tPrepared vertex buffer " << std::endl;
  //assert(false);

}

bool WriteToPly(const std::string &filename, const TriangleMesh &mesh) {
  (void)filename;
  (void)mesh;

  std::cerr << "Not yet implemented" << std::endl;

  // TODO(students): Implement storing to PLY format.

  // END.

  return false;
}

}  // namespace data_representation
