#include <Eigen/Dense>
#include <fstream>
#include <igl/bounding_box.h>
#include <igl/fast_winding_number.h>
#include <igl/list_to_matrix.h>
#include <igl/random_points_on_mesh.h>
#include <igl/readSTL.h>
#include <igl/signed_distance.h>
#include <igl/slice_mask.h>
#include <iostream>
#include <memory>

struct pointmesh
{
  Eigen::MatrixXd V;
  Eigen::MatrixXi F;
  Eigen::MatrixXd N;
  Eigen::MatrixXd Q;
  Eigen::VectorXf W;
  igl::FastWindingNumberBVH fwn_bvh;
  igl::AABB<Eigen::MatrixXd, 3> tree;
};

void
compute_bounding_box(const Eigen::MatrixXd& V,
                     Eigen::Vector3d& m,
                     Eigen::Vector3d& M,
                     Eigen::Vector3d& box_size)
{
  // Find the bounding box
  m = V.colwise().minCoeff();
  M = V.colwise().maxCoeff();
  box_size = M - m;
}

void
create_points(const Eigen::MatrixXd& V, int n, Eigen::MatrixXd& Q)
{
  Eigen::Vector3d m;
  Eigen::Vector3d M;
  Eigen::Vector3d box;
  compute_bounding_box(V, m, M, box);
  std::cout << "bounding_box:\n" << box << std::endl;

  Eigen::Vector3d box_step = box / n;
  Q = Eigen::MatrixXd::Zero(n * n * n, 3);
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
}

bool
read_points(const std::string& pts_filename, Eigen::MatrixXd& Q)
{
  std::ifstream ifs(pts_filename);

  if (!ifs.is_open())
    return false;

  std::string line;
  // count the number of line
  int n = 0;
  while (std::getline(ifs, line)) {
    n++;
  }

  // FIXME: Q.resize(n, 3) fails. Read data into a temporary array pts and
  // then copy into Q. Not efficient, but works.
  std::vector<std::vector<double>> pts;

  std::cout << "number of lines: " << n << std::endl;
  ifs.clear();
  ifs.seekg(0, ifs.beg);
  while (std::getline(ifs, line)) {
    std::stringstream ss(line);
    double x, y, z;
    ss >> x >> y >> z;
    pts.push_back({ x, y, z });
  }
  igl::list_to_matrix(pts, Q);
  return true;
}

void
compute_weights(struct pointmesh* ctx)
{
  // indexing
  ctx->tree.init(ctx->V, ctx->F);

  // igl::fast_winding_number(ctx->V.cast<float>(), ctx->F, 2, ctx->fwn_bvh);
  // compute
  // igl::fast_winding_number(ctx->fwn_bvh, 2, ctx->Q.cast<float>(), ctx->W);

  igl::SignedDistanceType sign_type = igl::SIGNED_DISTANCE_TYPE_PSEUDONORMAL;

  Eigen::VectorXi I;
  Eigen::MatrixXd C;
  Eigen::MatrixXd N;
  igl::signed_distance(ctx->Q, ctx->V, ctx->F, sign_type, ctx->W, I, C, N);
}

void
write_matrix(const std::string& out_filename, const Eigen::VectorXf& W)
{
  std::ofstream out(out_filename, std::ios::out | std::ios::trunc);
  for (int i = 0; i < W.rows(); ++i) {
    for (int j = 0; j < W.cols(); ++j) {
      out << W(i, j);
    }
    out << "\n";
  }
}

void
write_csv(const std::string& csv_filename,
          const Eigen::MatrixXd& Q,
          const Eigen::VectorXf& W)
{
  std::ofstream out(csv_filename, std::ios::out | std::ios::trunc);
  // header
  out << "x,y,z,val\n";
  for (int i = 0; i < W.rows(); ++i) {
    out << Q(i, 0) << "," << Q(i, 1) << "," << Q(i, 2);
    for (int j = 0; j < W.cols(); ++j) {
      out << "," << W(i, j);
    }
    out << "\n";
  }
}

bool
do_main(const std::string& cad_filename,
        const std::string& pts_filename,
        const std::string& out_filename,
        const int& sampling)
{

  std::unique_ptr<struct pointmesh> ctx(new struct pointmesh);
  igl::readSTL(cad_filename, ctx->V, ctx->F, ctx->N);

  if (sampling == 0) {
    read_points(pts_filename, ctx->Q);
  } else {
    create_points(ctx->V, sampling, ctx->Q);
  }
  std::cout << "points: " << ctx->Q.rows() << std::endl;
  compute_weights(ctx.get());
  write_matrix(out_filename, ctx->W);
  write_csv(out_filename + ".csv", ctx->Q, ctx->W);
  return true;
}

int
main(int argc, char** argv)
{
  std::string cad_filename;
  std::string pts_filename;
  std::string out_filename;
  int sampling = 0;

  for (int i = 1; i < argc; ++i) {
    if (strcmp(argv[i], "--cad") == 0 && (i + 1) < argc) {
      ++i;
      cad_filename = argv[i];
    } else if (strcmp(argv[i], "--points") == 0 && (i + 1) < argc) {
      ++i;
      pts_filename = argv[i];
    } else if (strcmp(argv[i], "--out") == 0 && (i + 1) < argc) {
      ++i;
      out_filename = argv[i];
    } else if (strcmp(argv[i], "--sampling") == 0 && (i + 1) < argc) {
      ++i;
      sampling = std::atoi(argv[i]);
      if (sampling <= 0) {
        std::cout << "error: bad sampling number: " << sampling << std::endl;
        return 1;
      }
    } else {
      std::cout << "unkown argument: " << argv[i] << std::endl;
      return 1;
    }
  }

  if (cad_filename.empty()) {
    std::cout << "error: missing arguments" << std::endl;
    return 1;
  }

  if (pts_filename.empty()) {
    pts_filename = cad_filename + ".pts";
    return 1;
  }

  if (out_filename.empty()) {
    out_filename = pts_filename + ".out";
  }

  return do_main(cad_filename, pts_filename, out_filename, sampling) ? 0 : 1;
}
