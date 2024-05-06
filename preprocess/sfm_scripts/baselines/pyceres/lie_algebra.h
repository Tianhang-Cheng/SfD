#include <Eigen/Dense>
#include <Eigen/Geometry>

#include "ceres/ceres.h"
#include "ceres/rotation.h"

#define Matrix4T Eigen::Matrix<T,4,4>
#define Matrix3T Eigen::Matrix<T,3,3>

#define Vector6T Eigen::Matrix<T,6,1>
#define Vector4T Eigen::Matrix<T,4,1>
#define Vector3T Eigen::Matrix<T,3,1>
#define Vector2T Eigen::Matrix<T,2,1>

constexpr double SINC_NUMERICAL_STABILITY_THRESHOLD = 1e-12;

template<typename T>
T stable_sinc(const T& x) {
    assert (x >= 0.0);
    if (x < T(SINC_NUMERICAL_STABILITY_THRESHOLD)) {
        return T(1.0);
    } else {
        return sin(x) / x;
    }
}

template<typename T>
T stable_cossinc(const T& x) {
    assert(x >= 0.0);
    if (x < T(SINC_NUMERICAL_STABILITY_THRESHOLD)) {
        return x;
    } else {
        return (T(1.0) - cos(x)) / x;
    }
}

template<typename T>
Matrix3T xprod_mat(const Vector3T& vec) {
    Matrix3T mat;
    mat << T(0.0), -vec(2), vec(1),
            vec(2), T(0.0), -vec(0),
            -vec(1), vec(0), T(0.0);
    return mat;
}

template<typename T>
Matrix3T jacobian_so3(const Vector3T & omega) {
    T angle = omega.norm();
    Vector3T axis = omega;
    if (angle >= T(SINC_NUMERICAL_STABILITY_THRESHOLD)) {
        axis = axis.normalized();
    } else {
        angle = T(0.0);
    }
    Matrix3T V = stable_sinc(angle) * Matrix3T::Identity();
    V += (T(1.0) - stable_sinc(angle)) * axis * axis.transpose();
    V += stable_cossinc(angle) * xprod_mat(axis);

    return V;
}

template<typename T>
Matrix3T inv_jacobian_so3(const Vector3T & omega) {
    T angle = omega.norm();

    assert(angle >= T(0.0));

    Vector3T axis = omega / angle;

    Matrix3T comp1, comp2, comp3;
    comp1.setIdentity();

    if (angle < T(SINC_NUMERICAL_STABILITY_THRESHOLD)) {
        comp2.setZero();
        comp3.setZero();
    } else {
        comp1 = comp1 * angle/2.0 * (1.0 / tan(angle/2.0));
        comp2 = (1.0 - angle/2.0 * (1.0 / tan(angle/2.0))) * axis * axis.transpose();
        comp3 = -angle/2.0 * xprod_mat(axis);
    }

    Matrix3T inv_jacobian = comp1 + comp2 + comp3;

    return inv_jacobian;
}

template<typename T>
Matrix4T exp_se3(const T* const xi) {
    T rot[9];
    ceres:AngleAxisToRotationMatrix<T>(xi, rot);
    Matrix3T R = Eigen::Map<Matrix3T>(rot);

    Vector3T omega;
    omega << xi[0], xi[1], xi[2];
    Matrix3T V = jacobian_so3(omega);
    Vector3T v;
    v << xi[3], xi[4], xi[5];
    Vector3T trans = V * v;
    
    Matrix4T pose = Matrix4T::Identity();
    pose.block(0,0,3,3) = R;
    pose.block(0,3,3,1) = trans;
    return pose;
}

template<typename T>
Vector6T ln_se3(Matrix4T P) {
    Matrix3T R = P.block(0,0,3,3);
    Vector3T t = P.block(0,3,3,1);

    T *R_arr = R.data();
    T omega_arr[3];
    ceres::RotationMatrixToAngleAxis(R_arr, omega_arr);
    Vector3T omega;
    omega << omega_arr[0], omega_arr[1], omega_arr[2];

    Matrix3T inv_so3_jacobian = inv_jacobian_so3(omega);

    Vector6T trans = inv_so3_jacobian * t;

    Vector6T xi_ret = Vector6T::Zero();
    xi_ret.block(0,0,3,1) = omega;
    xi_ret.block(3,0,3,1) = trans;
    return xi_ret;
}
