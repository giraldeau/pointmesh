#include <Eigen/Dense>
#include <igl/bounding_box.h>
#include <igl/fast_winding_number.h>
#include <igl/opengl/glfw/Viewer.h>
#include <igl/random_points_on_mesh.h>
#include <igl/readSTL.h>
#include <igl/slice_mask.h>
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

    int n = 50;
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
      igl::FastWindingNumberBVH fwn_bvh;
      igl::fast_winding_number(V.cast<float>(), F, 2, fwn_bvh);

      Eigen::VectorXf W;
      igl::fast_winding_number(fwn_bvh, 2, Q.cast<float>(), W);
      igl::slice_mask(Q, W.array() > 0.5, 1, Qinside);

      Eigen::VectorXf W2;
      W2.resize(Q.rows(), 1);
      int accuracy_scale = 2;
      float factor_inv = 1 / (4.0 * igl::PI);
      const double t_before = igl::get_seconds();
      for (int i = 0; i < Q.rows(); i++) {
        igl::FastWindingNumber::HDK_Sample::UT_Vector3T<float> Qp;
        Qp[0] = Q(i, 0);
        Qp[1] = Q(i, 1);
        Qp[2] = Q(i, 2);
        W2(i) = fwn_bvh.ut_solid_angle.computeSolidAngle(Qp, accuracy_scale) *
               factor_inv;
      }
      const double t_after = igl::get_seconds();
      const double delta = t_after - t_before;
      const double cost_per_pt = delta / Q.rows();
      std::cout << "computation time:" << delta << std::endl;
      std::cout << "computation time (per pt):" << cost_per_pt << std::endl;
    }

    std::cout << "Q total  " << Q.rows() << std::endl;
    std::cout << "Q inside " << Qinside.rows() << std::endl;

    int mesh_data = 0;
    int point_data = 1;

    viewer.data_list[mesh_data].set_points(V,
                                           Eigen::RowVector3d(1.0, 0.0, 0.0));
    viewer.data_list[mesh_data].point_size = 1;
    // viewer.data_list[mesh_data].set_face_based(true);

    viewer.append_mesh();
    viewer.data_list[point_data].set_points(Qinside,
                                            Eigen::RowVector3d(0.0, 1.0, 0.0));
    viewer.data_list[point_data].point_size = 2;
  }

  return viewer.launch();
  // return 0;
}
