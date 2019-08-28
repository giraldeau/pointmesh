#include <Eigen/Dense>
#include <igl/bounding_box.h>
#include <igl/fast_winding_number.h>
#include <igl/knn.h>
#include <igl/opengl/glfw/Viewer.h>
#include <igl/random_points_on_mesh.h>
#include <igl/readSTL.h>
#include <iostream>

template <typename DerivedV>
void bbox(Eigen::PlainObjectBase<DerivedV> &V, Eigen::Vector3d &m,
          Eigen::Vector3d &M) {
  // Find the bounding box
  m = V.colwise().minCoeff();
  M = V.colwise().maxCoeff();
}

int main(int argc, char *argv[]) {
  igl::opengl::glfw::Viewer viewer;

  if (argc >= 2) {
    std::string stl_file(argv[1]);

    Eigen::MatrixXd V;
    Eigen::MatrixXi F;
    Eigen::MatrixXd N;

    igl::readSTL(stl_file, V, F, N);

    Eigen::Vector3d m;
    Eigen::Vector3d M;
    Eigen::Vector3d box_size;
    bbox(V, m, M);
    box_size = M - m;

    std::cout << "m:\n" << m << std::endl;
    std::cout << "M:\n" << M << std::endl;
    std::cout << "s:\n" << box_size << std::endl;

    int n = 10;
    Eigen::Vector3d box_step = box_size / n;
    Eigen::MatrixXd Q = Eigen::MatrixXd::Zero(n * n * n, 3);
    int nrow = 0;
    for (int i = 0; i < n; i++) {
      for (int j = 0; j < n; j++) {
        for (int k = 0; k < n; k++) {
          Eigen::Vector3d pos(i, j, k);
          Q.row(nrow) = m.array() + pos.array() * box_step.array();
          nrow++;
        }
      }
    }


    Eigen::MatrixXd Qinside;
    {
      igl::FastWindingNumberBVH fwn;
    }


    std::cout << Q << std::endl;

    int mesh_data = 0;
    int point_data = 1;

    viewer.data_list[mesh_data].set_mesh(V, F);
    viewer.data_list[mesh_data].set_face_based(true);

    viewer.append_mesh();
    viewer.data_list[point_data].set_points(
        Q, Eigen::RowVector3d(0.996078, 0.760784, 0.760784));
  }

  return viewer.launch();
  //return 0;
}
