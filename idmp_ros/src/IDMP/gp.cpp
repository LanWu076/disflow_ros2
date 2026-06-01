/*
 *    This program is free software: you can redistribute it and/or modify
 *    it under the terms of the GNU General Public License v3 as published by
 *    the Free Software Foundation.
 *
 *    This program is distributed in the hope that it will be useful,
 *    but WITHOUT ANY WARRANTY; without even the implied warranty of
 *    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 *    GNU General Public License v3 for more details.
 *
 *    You should have received a copy of the GNU General Public License v3
 *    along with this program.  If not, see https://www.gnu.org/licenses/gpl-3.0.html.
 *
 *    Authors: Lan Wu <Lan.Wu-2@uts.edu.au>
 */

#include <IDMP/gp.h>
#include <Eigen/Cholesky>

#define SQRT_3  1.732051

namespace IDMP_ros {

#ifndef RevertingKernel
    inline float kf(float r, float a) {return (1.0+a*r)*exp(-a*r);}
    inline float kf1(float r, float dx, float a) {return a*a*dx*exp(-a*r);}
#else
    inline double kf(float r, float a){return exp(-r*r*a*0.5);}
    inline double kf1(float r, float dx,float a){return -dx*a*exp(-r*r*a*0.5);}
#endif

inline double kf_ma(double r, double a) {return (1.0+a*r)*exp(-a*r);} // matern kernel
inline double kf_ma1(double r, double dx, double a) {return a*a*dx*exp(-a*r);} // 1 derivative of se kernel
inline double kf_ma2(float r, float dx1, float dx2, float delta, float a){ // 2 derivative of se kernel
  return a*a*(delta-a*dx1*dx2/r)*exp(-a*r);}

    void gp::reset() {
        trained = false;
        return;
    }

    // 3D train
    EMatrixX gp::kernel_sparse_deriv1_3D(EMatrixX const& x1, float scale_param, float sigx)
    {
        int dim = x1.rows();
        int n = x1.cols();
        float a = scale_param;
        EMatrixX K = EMatrixX::Zero(n,n);
    
        for (int k=0;k<n;k++){
            for (int j=k;j<n;j++){
                if (k==j){
                    K(k,k) = 1.0+sigx;
                }
                else{
                    float r = (x1.col(k)-x1.col(j)).norm();
                    K(k,j) = kf(r,a);
                    K(j,k) = K(k,j);
                }
            }
        }
    
        return K;
    }

    void gp::train_new(const vecNode3 &samples){
        reset();
    
        int N = samples.size();
        int dim = 3;
    
        if (N > 0){
            x = EMatrixX::Zero(dim,N);
            EVectorX f = EVectorX::Zero(N);
            float sigx = 0.0001;
    
            int k=0;
            for (auto it = samples.begin(); it!=samples.end(); it++, k++){
                x(0,k) = (*it)->getPosX();
                x(1,k) = (*it)->getPosY();
                x(2,k) = (*it)->getPosZ();
                f(k) = 1;
            }
            
            EVectorX y(N);
            y << f;
            EMatrixX K = kernel_sparse_deriv1_3D(x, scale, sigx);
    
            L = K.llt().matrixL();
    
            alpha = y;
            L.template triangularView<Eigen::Lower>().solveInPlace(alpha);
            L.transpose().template triangularView<Eigen::Upper>().solveInPlace(alpha);
    
            trained = true;
    
        }
        return;
    }

    void gp::train_new(const Eigen::Matrix<double,3,Eigen::Dynamic> &samples){
        reset();
    
        int N = samples.cols();    
        if (N > 0){
            auto f = EVectorX::Ones(N);
            float sigx = 0.0001;

            x = samples;
            
            EMatrixX K = kernel_sparse_deriv1_3D(x, scale, sigx);
    
            L = K.llt().matrixL();
    
            alpha = f;
            L.template triangularView<Eigen::Lower>().solveInPlace(alpha);
            L.transpose().template triangularView<Eigen::Upper>().solveInPlace(alpha);
    
            trained = true;
        }
        //std::cout << "train_new"<< std::endl;
        return;
    }

    // 3D
    EMatrixX matern32_sparse_deriv1_3D(EMatrixX const& x1, std::vector<float> gradflag,
                                            float scale_param, EVectorX const& sigx, EVectorX const& siggrad)
    {
        int dim = x1.rows();
        int n = x1.cols();
        float sqr3L = sqrt(3)/scale_param;
        float sqr3L2 = sqr3L*sqr3L;
        EMatrixX K;

        int ng = 0;
        for (auto it = gradflag.begin();it!=gradflag.end();it++){
            if ((*it) > 0.5){
                (*it) = ng;
                ng++;
            }
            else
            {
                (*it) = -1.0;
            }
        }

        K = EMatrixX::Zero(n+ng*dim,n+ng*dim);

        for (int k=0;k<n;k++){
            int kind1=gradflag[k]+n;
            int kind2 = kind1+ng;
            int kind3 = kind2+ng;

            for (int j=k;j<n;j++){
                if (k==j){
                    K(k,k) = 1.0+sigx(k);
                    if (gradflag[k] > -0.5){
                        K(k,kind1) = 0.0;
                        K(kind1,k) = 0.0;
                        K(k,kind2) = 0.0;
                        K(kind2,k) = 0.0;
                        K(k,kind3) = 0.0;
                        K(kind3,k) = 0.0;

                        K(kind1,kind1) = sqr3L2+siggrad(k);;
                        K(kind1,kind2) = 0.0;
                        K(kind1,kind3) = 0.0;
                        K(kind2,kind1) = 0.0;
                        K(kind2,kind2) = sqr3L2+siggrad(k);
                        K(kind2,kind3) = 0.0;
                        K(kind3,kind1) = 0.0;
                        K(kind3,kind2) = 0.0;
                        K(kind3,kind3) = sqr3L2+siggrad(k);
                    }
                }
                else{
                    float r = (x1.col(k)-x1.col(j)).norm();
                    K(k,j) = kf_ma(r,sqr3L);
                    K(j,k) = K(k,j);
                    if (gradflag[k] > -1){

                        K(kind1,j) = -kf_ma1(r,x1(0,k)-x1(0,j),sqr3L);
                        K(j,kind1) = K(kind1,j);
                        K(kind2,j) = -kf_ma1(r,x1(1,k)-x1(1,j),sqr3L);
                        K(j,kind2) = K(kind2,j);
                        K(kind3,j) = -kf_ma1(r,x1(2,k)-x1(2,j),sqr3L);
                        K(j,kind3) = K(kind3,j);

                        if (gradflag[j] > -1){

                            int jind1=gradflag[j]+n;
                            int jind2 = jind1+ng;
                            int jind3 = jind2+ng;
                            K(k,jind1) = -K(j,kind1);
                            K(jind1,k) =  K(k,jind1);
                            K(k,jind2) = -K(j,kind2);
                            K(jind2,k) =  K(k,jind2);
                            K(k,jind3) = -K(j,kind3);
                            K(jind3,k) =  K(k,jind3);

                            K(kind1,jind1) = kf_ma2(r,x1(0,k)-x1(0,j),x1(0,k)-x1(0,j),1.0,sqr3L);
                            K(jind1,kind1) = K(kind1,jind1);
                            K(kind1,jind2) = kf_ma2(r,x1(0,k)-x1(0,j),x1(1,k)-x1(1,j),0.0,sqr3L);
                            K(jind1,kind2) = K(kind1,jind2);
                            K(kind1,jind3) = kf_ma2(r,x1(0,k)-x1(0,j),x1(2,k)-x1(2,j),0.0,sqr3L);
                            K(jind1,kind3) = K(kind1,jind3);

                            K(kind2,jind1) = K(kind1,jind2);
                            K(jind2,kind1) = K(kind1,jind2);
                            K(kind2,jind2) = kf_ma2(r,x1(1,k)-x1(1,j),x1(1,k)-x1(1,j),1.0,sqr3L);
                            K(jind2,kind2) = K(kind2,jind2);
                            K(kind2,jind3) = kf_ma2(r,x1(1,k)-x1(1,j),x1(2,k)-x1(2,j),0.0,sqr3L);
                            K(jind2,kind3) = K(kind2,jind3);

                            K(kind3,jind1) = K(kind1,jind3);
                            K(jind3,kind1) = K(kind1,jind3);
                            K(kind3,jind2) = K(kind2,jind3);
                            K(jind3,kind2) = K(kind2,jind3);
                            K(kind3,jind3) = kf_ma2(r,x1(2,k)-x1(2,j),x1(2,k)-x1(2,j),1.0,sqr3L);
                            K(jind3,kind3) = K(kind3,jind3);
                        }
                    }
                    else if (gradflag[j] > -1){

                        int jind1=gradflag[j]+n;
                        int jind2 = jind1+ng;
                        int jind3 = jind2+ng;
                        K(k,jind1) = kf_ma1(r,x1(0,k)-x1(0,j),sqr3L);
                        K(jind1,k) = K(k,jind1);
                        K(k,jind2) = kf_ma1(r,x1(1,k)-x1(1,j),sqr3L);
                        K(jind2,k) = K(k,jind2);
                        K(k,jind3) = kf_ma1(r,x1(2,k)-x1(2,j),sqr3L);
                        K(jind3,k) = K(k,jind3);
                    }
                }
            }
        }

        return K;
    }

    EMatrixX matern32_sparse_deriv1_3D(EMatrixX const& x1, std::vector<float> gradflag,
                                            EMatrixX const& x2, float scale_param)
    {
        int dim = x1.rows();
        int n = x1.cols();
        float sqr3L = sqrt(3)/scale_param;
        float sqr3L2 = sqr3L*sqr3L;
        EMatrixX K;

        int ng = 0;
        for (auto it = gradflag.begin();it!=gradflag.end();it++){
            if ((*it) > 0.5){
                (*it) = ng;
                ng++;
            }
            else
            {
                (*it) = -1.0;
            }
        }

        int m = x2.cols();
        int m2 = m+m;
        int m3 = m2+m;

        K = EMatrixX::Zero(n+ng*dim,m*(1+dim));

        for (int k=0;k<n;k++){
            int kind1=gradflag[k]+n;
            int kind2 = kind1+ng;
            int kind3 = kind2+ng;
            for (int j=0;j<m;j++){
                float r = (x1.col(k)-x2.col(j)).norm();

                K(k,j) = kf_ma(r,sqr3L);
                K(k,j+m) = kf_ma1(r,x1(0,k)-x2(0,j),sqr3L);
                K(k,j+m2) = kf_ma1(r,x1(1,k)-x2(1,j),sqr3L);
                K(k,j+m3) = kf_ma1(r,x1(2,k)-x2(2,j),sqr3L);
                if (gradflag[k] > -0.5){
                    K(kind1,j) = -K(k,j+m);
                    K(kind2,j) = -K(k,j+m2);
                    K(kind3,j) = -K(k,j+m3);
                    K(kind1,j+m) = kf_ma2(r,x1(0,k)-x2(0,j),x1(0,k)-x2(0,j),1.0,sqr3L);
                    K(kind1,j+m2) =  kf_ma2(r,x1(0,k)-x2(0,j),x1(1,k)-x2(1,j),0.0,sqr3L);
                    K(kind1,j+m3) =  kf_ma2(r,x1(0,k)-x2(0,j),x1(2,k)-x2(2,j),0.0,sqr3L);
                    K(kind2,j+m) = K(kind1,j+m2);
                    K(kind2,j+m2) = kf_ma2(r,x1(1,k)-x2(1,j),x1(1,k)-x2(1,j),1.0,sqr3L);
                    K(kind2,j+m3) = kf_ma2(r,x1(1,k)-x2(1,j),x1(2,k)-x2(2,j),0.0,sqr3L);
                    K(kind3,j+m) = K(kind1,j+m3);
                    K(kind3,j+m2) = K(kind2,j+m3);
                    K(kind3,j+m3) = kf_ma2(r,x1(2,k)-x2(2,j),x1(2,k)-x2(2,j),1.0,sqr3L);
                }
            }
        }

        return K;
    }

    void gp::train_new(const Eigen::Matrix<double,3,Eigen::Dynamic> &samples,
                    const Eigen::Matrix<double,3,Eigen::Dynamic> &normals)
    {
        reset();
        //std::cout << "train_new: "<< std::endl;
        const int dim = 3;
        const int N   = samples.cols();
        
        if (samples.rows() != 3 || normals.rows() != 3 || normals.cols() != N) {
            throw std::runtime_error("gp::train_new: samples/normals must be 3xN with matching N.");
        }

        if (N > 0) {
            // Store training inputs in your 3xN layout
            x = samples.cast<double>();                 // 3×N
            const EMatrixX grad = normals.cast<double>(); // 3×N

            // Hyperparams / nuggets
            const double sig_f = 1e-4;                  // function noise
            const double sig_g = 1e-3;                  // gradient noise
            const double n_eps = 1e-6;                  // normal validity threshold

            // Per-point flags and noises
            gradflag.clear();
            gradflag.resize(N,0.0);
            EVectorX sigx    = EVectorX::Constant(N, sig_f);
            EVectorX siggrad = EVectorX::Zero(N);       // only used for valid ones

            // Build valid gradient index order (must match kernel's packing order)
            std::vector<double> nx_v; nx_v.reserve(N);
            std::vector<double> ny_v; ny_v.reserve(N);
            std::vector<double> nz_v; nz_v.reserve(N);

            for (int k = 0; k < N; ++k) {
                const double nx = grad(0,k), ny = grad(1,k), nz = grad(2,k);
                const double nrm = std::sqrt(nx*nx + ny*ny + nz*nz);
                if (nrm > n_eps) {
                    gradflag[k] = 1.0f;                 // mark as valid; kernel will remap to 0..ng-1
                    siggrad(k)  = sig_g;
                    nx_v.push_back(nx);
                    ny_v.push_back(ny);
                    nz_v.push_back(nz);
                } else {
                    gradflag[k] = 0.0f;
                    // Optionally inflate function noise for points w/o gradient:
                    // sigx(k) = 2.0;  // keep or drop based on your preference
                }
            }
            const int G = static_cast<int>(nx_v.size()); // number of valid gradient points

            // Targets: f first, then all nx(valid), then ny(valid), then nz(valid)
            EVectorX y(N + 3*G);
            // If you want bias GP on-surface, set all ones:
            // y.head(N).setOnes();
            // Or use external f-values if you have them; here we use ones:
            y.head(N).setZero();

            //std::cout << "train_new with normal"<< std::endl;

            for (int i = 0; i < G; ++i) y(N + i)       = nx_v[i];
            for (int i = 0; i < G; ++i) y(N + G + i)   = ny_v[i];
            for (int i = 0; i < G; ++i) y(N + 2*G + i) = nz_v[i];

            // Build full (N + 3G) × (N + 3G) kernel using your packed kernel builder
            // NOTE: param.scale is your length-scale ℓ (your kernel uses a = sqrt(3)/ℓ internally)
            EMatrixX K = matern32_sparse_deriv1_3D(x, gradflag, 0.03, sigx, siggrad);

            // Solve K alpha = y via LLT
            Eigen::LLT<EMatrixX> llt(K);
            if (llt.info() != Eigen::Success) {
                throw std::runtime_error("gp::train_new: Cholesky failed (ill-conditioned K).");
            }
            L = llt.matrixL();

            alpha = y;
            L.template triangularView<Eigen::Lower>().solveInPlace(alpha);
            L.transpose().template triangularView<Eigen::Upper>().solveInPlace(alpha);

            trained = true;
        }
        return;
    }

    //Test matern
    EMatrixX gp::kernel_sparse_deriv1_3D(EMatrixX const& x1, EMatrixX const& x2, float scale_param)
    {
        int dim = x1.rows();
        int n = x1.cols();
        int m = x2.cols();
        float a = scale_param;
        EMatrixX K = EMatrixX::Zero(n,m);
    
        for (int k=0;k<n;k++){
            for (int j=0;j<m;j++){
                float r = (x1.col(k)-x2.col(j)).norm();
                K(k,j) = kf(r,a);
            }
        }
    
        return K;
    }

    void gp::testSinglePointG(const EVectorX& xt, double& val, double grad[],double var[])
    {
        //std::cout << "testSinglePointGs"<< std::endl;
        if (!isTrained())
            return;

        if (x.rows() != xt.size())
            return;

        EMatrixX K = matern32_sparse_deriv1_3D(x, gradflag, xt, 0.03);

        EVectorX res = K.transpose()*alpha;
        val = res(0);
        if (res.size() == 3){
            grad[0] = res(1);
            grad[1] = res(2);
        }
        else if (res.size() == 4){
            grad[0] = res(1);
            grad[1] = res(2);
            grad[2] = res(3);
        }

        L.template triangularView<Eigen::Lower>().solveInPlace(K);
        K = K.array().pow(2);
        EVectorX v = K.colwise().sum();

        if (v.size() == 3){
            var[0] = 1.01-v(0);
            var[1] = three_over_scale + 0.1 - v(1);
            var[2] = three_over_scale + 0.1 - v(2);
        }
        else if (v.size() == 4){ // Noise param!
            var[0] = 1.001-v(0);
            var[1] = three_over_scale + 0.001 - v(1);
            var[2] = three_over_scale + 0.001 - v(2);
            var[3] = three_over_scale + 0.001 - v(3);
        }
        //std::cout << "testSinglePointGs"<< std::endl;

        return;
    }

    void gp::testSinglePoint_new(const EVectorX& xt, double& val, double grad[], double var[]) {
        if (!isTrained())
            return;
    
        if (x.rows() != xt.size())
            return;
    
        EMatrixX K = kernel_sparse_deriv1_3D(x, xt, scale);
        EVectorX res = K.transpose()*alpha;
        val = res(0);
    
        int n = x.cols();
        int m = xt.cols();    
        float a = scale;
    
        EMatrixX Grx = EMatrixX::Zero(n,m);
        EMatrixX Gry = EMatrixX::Zero(n,m);
        EMatrixX Grz = EMatrixX::Zero(n,m);
    
        for (int k=0;k<n;k++){
            for (int j=0;j<m;j++){
                float r = (x.col(k)-xt.col(j)).norm();
                Grx(k, j) = kf1(r,x(0, k) - xt(0, j),a);
                Gry(k, j) = kf1(r,x(1, k) - xt(1, j),a);
                Grz(k, j) = kf1(r,x(2, k) - xt(2, j),a);
            }
        }
    
        EVectorX gradx = Grx.transpose()*alpha;
        EVectorX grady = Gry.transpose()*alpha;
        EVectorX gradz = Grz.transpose()*alpha;

        grad[0] = gradx(0);  
        grad[1] = grady(0);
        grad[2] = gradz(0);
    
        return;
    }
}
