#include <iostream>
#include <vector>

#include <boost/python.hpp>
#include <boost/python/tuple.hpp>
#include <boost/python/list.hpp>

#include <Eigen/Core>
#include "ceres/ceres.h"
#include "ceres/rotation.h"

using namespace std;
using namespace ceres;

namespace py = boost::python;


struct ReprojectionResidual {
    ReprojectionResidual(const vector<double> & pt_2d, const Eigen::Matrix3d & cam_intrinsics)
        : pt_2d(pt_2d), cam_intrinsics(cam_intrinsics) {}

    template <typename T>
    bool operator()(const T* const xi_ext, const T* const pt_3d, T* residual) const {
        // camera[0,1,2] are the angle-axis rotation.
        T p[3];
        AngleAxisRotatePoint(xi_ext, pt_3d, p);
        
        // camera[3,4,5] are the translation.
        p[0] += xi_ext[3]; p[1] += xi_ext[4]; p[2] += xi_ext[5];
      
        // projection
        T projected_u = T(cam_intrinsics(0,0)) * p[0] / p[2] + T(cam_intrinsics(0,2));
        T projected_v = T(cam_intrinsics(1,1)) * p[1] / p[2] + T(cam_intrinsics(1,2));

        residual[0] = T(pt_2d[0]) - projected_u;
        residual[1] = T(pt_2d[1]) - projected_v;

        return true;
    }

    // Factory to hide the construction of the CostFunction object from
    // the client code.
    static CostFunction* Create(
        const vector<double> & pt_2d, const Eigen::Matrix3d & cam_intrinsics) {
            return (
                new AutoDiffCostFunction<ReprojectionResidual, 2, 6, 3>(
                    new ReprojectionResidual(pt_2d, cam_intrinsics)));
    }
 
    private:
        // Observations for a sample.
        const vector<double> pt_2d;
        Eigen::Matrix3d cam_intrinsics;
};


struct PointRegResidual {
    PointRegResidual(const vector<double> & pt_3d_dist, const double & weight)
        : pt_3d_dist(pt_3d_dist), weight(weight) {}

    template <typename T>
    bool operator()(const T* const pt1, const T* const pt2, T* residual) const {
        for (unsigned coord = 0; coord < 3; ++coord) {
            residual[coord] = weight * ((pt1[coord] - pt2[coord]) - T(pt_3d_dist[coord]));
        }

        return true;
    }

    // Factory to hide the construction of the CostFunction object from
    // the client code.
    static CostFunction* Create(
        const vector<double> & pt_3d_dist, const double & weight) {
            return (
                new AutoDiffCostFunction<PointRegResidual, 3, 3, 3>(
                    new PointRegResidual(pt_3d_dist, weight)));
    }
 
    private:
        const vector<double> pt_3d_dist;
        const double weight;
};


void bundle_adjustment(
    const vector<vector<double>> pts_2d,
    const vector<int> pts_id,
    const Eigen::Matrix3d & cam_intrinsics,
    vector<vector<double>> & pts_3d,
    vector<double> & xi_ext
) {
    Problem problem;
    assert(pts_2d.size() == pts_id.size());
    for (unsigned idx = 0; idx < pts_2d.size(); ++idx) {
        CostFunction* cost_function = ReprojectionResidual::Create(pts_2d[idx], cam_intrinsics);
        problem.AddResidualBlock(cost_function, new HuberLoss(10.0), xi_ext.data(), pts_3d[pts_id[idx]].data());
        // problem.SetParameterBlockConstant(pts_3d[pts_id[idx]].data());
    }

    for (unsigned i = 0; i < pts_3d.size(); ++i) {
        for (unsigned j = i + 1; j < pts_3d.size(); ++j) {
            vector<double> pt_3d_dist;
            for (unsigned coord = 0; coord < 3; ++coord) {
                pt_3d_dist.emplace_back(pts_3d[i][coord] - pts_3d[j][coord]);
            }
            CostFunction* cost_function = PointRegResidual::Create(pt_3d_dist, 1000.0);
            problem.AddResidualBlock(cost_function, NULL, pts_3d[i].data(), pts_3d[j].data());
        }
    }


    // Run the solver!
    Solver::Options options;
    options.linear_solver_type = DENSE_QR;
    options.minimizer_progress_to_stdout = true;
    // options.function_tolerance = 1e-10;
    options.max_num_iterations = 2000;
    Solver::Summary summary;
    Solve(options, &problem, &summary);

    std::cout << summary.FullReport() << "\n";
    // std::cout << summary.BriefReport() << "\n";
}


py::tuple py_bundle_adjustment(
    const py::list & observations,
    const py::list & K,
    const py::list & points,
    const py::list & T_init
) {
    // extract observations
    vector<vector<double>> pts_2d;
    vector<int> pts_id;
    int num_obs = py::len(observations);
    pts_2d.reserve(num_obs);
    pts_id.reserve(num_obs);
    for (unsigned idx = 0; idx < num_obs; ++idx) {
        py::list obs = py::extract<py::list>(observations[idx]);
        assert(py::len(obs) == 3);
        double u = py::extract<double>(obs[0]);
        double v = py::extract<double>(obs[1]);
        int pt_id = py::extract<int>(obs[2]);
        vector<double> uv(u, v);
        pts_2d.emplace_back(vector<double>{u, v});
        pts_id.emplace_back(pt_id);
    }

    // extract 3d points
    vector<vector<double>> pts_3d;
    int num_pts = py::len(points);
    pts_3d.reserve(num_pts);
    for (unsigned idx = 0; idx < num_pts; ++idx) {
        py::list pt = py::extract<py::list>(points[idx]);
        assert(py::len(pt) == 3);
        vector<double> pt_3d;
        for (unsigned coord = 0; coord < 3; ++coord) {
            double val = py::extract<double>(pt[coord]);
            pt_3d.emplace_back(val);
        }
        pts_3d.emplace_back(pt_3d);
    }

    // extract cam_intrinsics
    Eigen::Matrix3d cam_intrinsics;
    assert(py::len(K) == 3);
    for (unsigned idx = 0; idx < 3; ++idx) {
        py::list row = py::extract<py::list>(K[idx]);
        assert(py::len(row) == 3);
        for (unsigned coord = 0; coord < 3; ++coord) {
            double val = py::extract<double>(row[coord]);
            cam_intrinsics(idx, coord) = val;
        }
    }
    // cout << "K " << cam_intrinsics << endl;

    // extract xi_ext
    Eigen::Matrix4d pose_mat = Eigen::Matrix4d::Identity();
    assert(py::len(T_init) == 4);
    for (unsigned idx = 0; idx < 4; ++idx) {
        py::list row = py::extract<py::list>(T_init[idx]);
        assert(py::len(row) == 4);
        for (unsigned coord = 0; coord < 4; ++coord) {
            double val = py::extract<double>(row[coord]);
            pose_mat(idx, coord) = val;
        }
    }
    // cout << "Pose Mat " << pose_mat << endl;
    double omega_arr[3];
    Eigen::Matrix3d rot_mat = pose_mat.block(0,0,3,3);
    double *R_arr = rot_mat.data();
    RotationMatrixToAngleAxis(R_arr, omega_arr);
    // cout << "omega array " << omega_arr[0] << " " << omega_arr[1] << " " << omega_arr[2] << endl;
    vector<double> xi_ext;
    xi_ext.reserve(6);
    for (unsigned idx = 0; idx < 3; ++idx) {
        xi_ext.emplace_back(omega_arr[idx]);
    }
    for (unsigned idx = 0; idx < 3; ++idx) {
        xi_ext.emplace_back(pose_mat(idx,3));
    }

    // std::cout << "xi_init" << std::endl;
    // for (const double & val : xi_ext) {
    //     std::cout << val << std::endl;
    // }

    bundle_adjustment(pts_2d, pts_id, cam_intrinsics, pts_3d, xi_ext);

    // prepare output
    double rot_out[9];
    AngleAxisToRotationMatrix(xi_ext.data(), rot_out);
    py::list T_out;
    py::list pts_out;
    for (unsigned idx = 0; idx < 3; ++idx) {
        py::list row;
        for (unsigned coord = 0; coord < 3; ++coord) {
            row.append(rot_out[idx * 3 + coord]);
        }
        row.append(xi_ext[idx+3]);
        T_out.append(row);
    }
    py::list last_row;
    last_row.append(0); last_row.append(0); last_row.append(0); last_row.append(1);
    T_out.append(last_row);

    for (const vector<double> & pt : pts_3d) {
        py::list pt_list;
        for (const double & val : pt) {
            pt_list.append(val);
        }
        pts_out.append(pt_list);
    }

    return py::make_tuple(T_out, pts_out);
}


BOOST_PYTHON_MODULE(libpyceres_ba) {
    // Py_Initialize();
    py::def("ceres_ba", py_bundle_adjustment);
}