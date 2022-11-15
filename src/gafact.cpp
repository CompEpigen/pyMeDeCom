#include <eigen3/Eigen/LU>
#include <eigen3/Eigen/Dense>

using Eigen::Dynamic;
using PyIn  = Eigen::Ref<Eigen::MatrixXd>;


class ProjGradT {
    double eps;

 public:
    explicit ProjGradT(double tol) : eps(tol)
        {}

        inline double operator()(const double& g, const double& t) const {
            if (t <= eps) {
                return std::min(g, 0.0);
            } else if (eps < t && t < 1 - eps) {
                return g;
            } else {
                return std::max(g, 0.0);
            }
        }
};


template <typename MatrixBig, int DIM = 16, typename Scalar = double>
class ProbSimplexProjector {
 public:
    using Matrix = Eigen::Matrix<Scalar, DIM, Dynamic>;

 private:
    const Matrix mTtD;
    const Matrix mTtT;
    double tol;
    int itersMax;

    int niter;
    double optCond;

    int r;
    int n;

 public:
    ProbSimplexProjector(const MatrixBig& Dt, const Matrix& Tt, double tol,
        int itersMax) : mTtD(Tt * Dt.transpose()), mTtT(Tt * Tt.transpose()),
        tol(tol), itersMax(itersMax), r(Tt.rows()), n(Dt.rows())
    {}

    void solve(Matrix& mA) {
        /* init */
        niter   = 1;
        optCond = 1e+10;

        double cL = mTtT.operatorNorm() + tol;
        double lrA = 1.0 / cL;

        Matrix mAy = mA;
        Matrix mAnext = Matrix::Zero(r, n);
        Matrix gradA  = Matrix::Zero(r, n);
        double tcurr = 1.0, tnext = 1.0;

        while (niter <= itersMax && optCond > tol) {
            evalGrad(mAy, gradA);
            mAnext = mAy - lrA * gradA;
            colwiseProjProbSplx(mAnext);

            /* Check for restart. Gradient-mapping based test */
            if ((cL * (mAy - mAnext)).cwiseProduct(mAnext - mA).sum() > 0) {
                /* Restart */
                mAy = mAnext;
            } else {
                mAy = mAnext + (tcurr - 1.0) / tnext * (mAnext - mA);
            }

            tnext = 0.5 * (1 + std::sqrt(1 + 4 * tcurr * tcurr));

            /* Stopping criteria */
            ++niter;
            optCond = (mAnext - mA).norm();

            mA = mAnext;
            tcurr = tnext;
        }
    }

 private:
    void evalGrad(const Matrix& A, Matrix& grad) {
        grad = mTtT * A - mTtD;
    }

    void colwiseProjProbSplx(Matrix& mA) {
        int n = mA.cols();
        int r = mA.rows();
        Matrix mAcopy = mA;

        for (int colN = 0; colN < n; ++colN) {
            auto s = mAcopy.col(colN);
            std::sort(s.data(), s.data() + s.size(),
                    std::greater<double>());
            bool bget = false;
            double tmpsum = 0.0, tmax;
            for (int ii = 0; ii < r - 1; ++ii) {
                tmpsum += s(ii);
                tmax = (tmpsum - 1.0) / (ii + 1);
                if (tmax >= s(ii + 1)) {
                    bget = true;
                    break;
                }
            }

            if (!bget) {
                tmax = (tmpsum + s(r - 1) - 1) / r;
            }

            for (int jj = 0; jj < r; ++jj) {
                mA(jj, colN) = std::max(mA(jj, colN) - tmax, 0.0);
            }
        }
    }
};


template <int DIM = 16, typename Scalar = double>
class CoordDescentSolver {
 public:
    using Matrix = Eigen::Matrix<Scalar, DIM, DIM>;
    using Array  = Eigen::Array<Scalar, DIM, 1>;
    using Vector = Eigen::Matrix<Scalar, DIM, 1>;

    CoordDescentSolver(double tol, int itersMax)
        : tol(tol), itersMax(itersMax)
    {}
    void solve(const Matrix& AAt, Vector& tinit, const Vector& b) {
        tinit = tinit.cwiseMax(0.0).cwiseMin(1.0);
        solveCoordDescent(AAt, tinit, b);
    }

 private:
    double tol;
    int itersMax;

    const Scalar slackEps = 1e-15;

    void solveCoordDescent(const Matrix& AAt, Vector& t, const Vector& b) {
        double optCond = tol + 1.0; int niter = 1;
        int r = AAt.cols();

        Vector grad;
        Scalar prod;
        while (niter <= itersMax && optCond > tol) {
            grad = -b;
            for (int i = 0; i < r; ++i) {
                prod = AAt.col(i).dot(t) - b(i);
                t(i) -= prod / AAt(i, i);
                t(i) = std::max(0.0, t(i));
                t(i) = std::min(1.0, t(i));
                grad.noalias() += AAt.col(i) * t(i);
            }
            /* Evaluate optimality condition */
            optCond = grad.binaryExpr(t, ProjGradT(slackEps)).norm();
            /* finish this iteration */
            ++niter;
        }
    }
};

struct SolverSuppOutput {
    int niters;
    double objF;
    double rmse;
};

template <int DIM = -1>
void applySolver(const PyIn& Dt, PyIn& Tt0, PyIn& A0,
        double lambda, int itersMax, int innerItersMax, double tol,
        double tolA, double tolT, SolverSuppOutput& supp) {
    using MatrixDD = Eigen::Matrix<double, DIM, DIM>;
    using MatrixDX = Eigen::Matrix<double, DIM, Dynamic>;
    using Vector   = Eigen::Matrix<double, DIM, 1>;

    int n = Dt.rows();
    int m = Dt.cols();
    int k = A0.rows();

    /* Convert to Eigen's data types */
    MatrixDX Tt = Tt0.template cast<double>();
    MatrixDX A  = A0.template cast<double>();

    /* Time-savers */
    auto onesrm = MatrixDX::Ones(k, m);

    int niter = 1;
    double optCond = 1e+10;

    double dA, dT;
    MatrixDX Ttprev, Aprev;
    while (niter <= itersMax && optCond > tol) {
        Ttprev = Tt;
        Aprev = A;

        // Optimize A
        ProbSimplexProjector<PyIn, DIM> probSmplxProjector(
            Dt, Tt, tolA, innerItersMax);
        probSmplxProjector.solve(A);

        // Optimize T
        MatrixDD AAt = A * A.transpose();
        MatrixDX B = A * Dt - lambda * (onesrm - 2 * Ttprev);

        CoordDescentSolver<DIM> solver(tolT, innerItersMax);
        for (int i = 0; i < m; ++i) {
            Vector t = Tt.col(i);
            Vector b = B.col(i);
            solver.solve(AAt, t, b);
            Tt.col(i) = t;
        }
        ++niter;
        dA = (Aprev - A).norm() / std::sqrt(k * n);
        dT = (Ttprev - Tt).norm() / std::sqrt(m * k);
        optCond = std::sqrt(dA * dA + dT * dT);
    }
    // Assign solutions to python references
    A0 = A; Tt0 = Tt;
    supp.niters = niter - 1;
    supp.rmse   = 0.5 * (Dt - A.transpose() * Tt).squaredNorm();
    supp.objF   = supp.rmse + lambda * (Tt.sum() - Tt.squaredNorm());
    supp.rmse  /= m;
    supp.rmse  /= n;
}

/* Some Voodoo magic to eliminate
 * long switches for different dimensions */
template <int ...> struct DimList {};

/* border case */
void solve(int d, const PyIn& mDt, PyIn& mTtinit, PyIn& mAinit,
        double lambda, int itersMax, int innerItersMax, double tol, double tolA,
        double tolT, SolverSuppOutput& supp, DimList<>) {
}

template <int DIM, int ...DIMS>
void solve(int d, const PyIn& mDt, PyIn& mTtinit, PyIn& mAinit,
        double lambda, int itersMax, int innerItersMax, double tol, double tolA,
        double tolT, SolverSuppOutput& supp, DimList<DIM, DIMS...>) {
    if (DIM != d) {
        return solve(d, mDt, mTtinit, mAinit, lambda,
                itersMax, innerItersMax, tol, tolA, tolT,
                supp,
                DimList<DIMS...>());
    }

    applySolver<DIM>(mDt, mTtinit, mAinit, lambda,
            itersMax, innerItersMax, tol, tolA, tolT,
            supp);
}

template <int ...DIMS>
void solve(int d, const PyIn& mDt, PyIn& mTtinit, PyIn& mAinit,
        double lambda, int itersMax, int innerItersMax, double tol, double tolA,
        double tolT, SolverSuppOutput& supp) {
        solve(d, mDt, mTtinit, mAinit, lambda,
                itersMax, innerItersMax, tol, tolA, tolT,
                supp,
                DimList<DIMS...>());
}

SolverSuppOutput TAFact(
    PyIn D, PyIn Tt0, PyIn A0,
    double lmbda, int itersMax, int innerItersMax,
    double tol, double tolA, double tolT){

    Eigen::initParallel();
    Eigen::setNbThreads(1);
    SolverSuppOutput supp;

    solve<Dynamic>(
        Dynamic, D, Tt0, A0,
        lmbda, itersMax, innerItersMax, tol,
        tolA, tolT, supp);
    return supp;
}


