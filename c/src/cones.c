/*
*  CPSR - Cone Program Solution Refinement
*
*  Copyright (C) 2019, Enzo Busseti
*
*  This program is free software: you can redistribute it and/or modify
*  it under the terms of the GNU General Public License as published by
*  the Free Software Foundation, either version 3 of the License, or
*  (at your option) any later version.
*
*  This program is distributed in the hope that it will be useful,
*  but WITHOUT ANY WARRANTY; without even the implied warranty of
*  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
*  GNU General Public License for more details.
*
*  You should have received a copy of the GNU General Public License
*  along with this program.  If not, see <https://www.gnu.org/licenses/>.
*/

#include <cones.h>
#include <math.h>
#include <string.h>
#include <stdbool.h>
#include <cblas.h>


// double norm(double *x, int size){
    
//     double norm_x = 0.;
//     for (int i = 0; i < size; i++){
//         norm_x += x[i] * x[i];
//     }
//     return sqrt(norm_x);
// }



// void vecsum(double *x, double *y, int size){
//     for (int i = 0; i < size; i++){
//         x[i] += y[i];
//     }
// }

// void vecdiff(double *x, double *y, int size){
//     for (int i = 0; i < size; i++){
//         x[i] -= y[i];
//     }
// }

// void vecalgsum(double *x, double *y, 
//                  double alpha, double beta,
//                  int size){
//     for (int i = 0; i < size; i++){
//         x[i] = alpha * x[i] + beta * y[i];
//     }
// }

// double dot(double *x, double *y, int size){
//     double result = 0.;
//     for (int i = 0; i < size; i++){
//         result += x[i] * y[i];
//     }
//     return result;
// }



void zero_cone_projection(double *z, const vecsize size){
    memset(z, 0, sizeof(double) * size);
}

void zero_cone_projection_derivative(const double *z, double *dz, 
                                     const vecsize size){
    memset(dz, 0, sizeof(double) * size);
}

void non_negative_cone_projection(double *z, const vecsize size){
    for (int i = 0; i < size; i++){
        if (z[i] < 0) {
            z[i] = 0;
        };
    }
}

void non_negative_cone_projection_derivative(const double *z, double *dz, const vecsize size){
    for (int i = 0; i < size; i++){
        if (z[i] < 0) {
            dz[i] = 0;
        };
    }
}

void second_order_cone_projection(double *z, const vecsize size){

    double norm_x = cblas_dnrm2(size - 1, z + 1, 1);


    if (norm_x <= z[0]){
        return;
    }

    if (norm_x <= -z[0]){
        memset(z, 0, sizeof(double) * size);
        return;
    }

    double rho = z[0];

    z[0] = (norm_x + rho) / 2.;

    double mult = z[0]/norm_x;

    for (int i = 1; i < size; i++){
        z[i] *= mult;
    } 
}


void second_order_cone_projection_derivative(const double *z, 
                                             double *dz, 
                                             const double *pi_z,
                                             const vecsize size){

    // point at which we derive
    // double t = z[0];
    // x = z[1:]

    // projection of point
    // s = pi_z[0]
    // y = pi_z[1:]

    // logic for simple cases
    //norm = np.linalg.norm(z[1:])

    double norm_x = cblas_dnrm2(size - 1, z + 1, 1);

    if (norm_x <= z[0]){return;}
    
    if (norm_x <= -z[0]){
        memset(dz, 0, sizeof(double) * size);
        return;
    }

    // big case
    double dot_val = cblas_ddot(size - 1, dz + 1, 1, z + 1, 1);

    double old_dzzero = dz[0];
    
    dz[0] = (dz[0] * norm_x + dot_val) / (2 * norm_x);

    double first_coefficient = (z[0] + norm_x) / (2. * norm_x);
    double second_coefficient = (old_dzzero - z[0] * dot_val / (norm_x*norm_x)) / (2. * norm_x);

    cblas_dscal(size - 1, first_coefficient, dz + 1, 1);
    cblas_daxpy(size - 1, second_coefficient, z + 1, 1, dz + 1, 1);

    return;

}


#define CONE_TOL (1e-16)
#define CONE_THRESH (1e-16)
#define CONE_THRESH_TWO (1e-16)
#define EXP_CONE_MAX_ITERS (200)

const double EulerConstant = 2.718281828459045; //exp(1.0);

double exp_newton_one_d(double rho, double y_hat, double z_hat) {
  double t = (-z_hat > CONE_THRESH) ? -z_hat : CONE_THRESH;
  double f, fp;
  int i;
  for (i = 0; i < EXP_CONE_MAX_ITERS; ++i) {
    f = t * (t + z_hat) / rho / rho - y_hat / rho + log(t / rho) + 1;
    fp = (2 * t + z_hat) / rho / rho + 1 / t;

    t = t - f / fp;

    if (t <= -z_hat) {
      return 0;
    } else if (t <= 0) {
      return z_hat;
    } else if (fabs(f) < CONE_TOL) {
      break;
    }
  }
  return t + z_hat;
}

void exp_solve_for_x_with_rho(double *v, double *x, double rho) {
  x[2] = exp_newton_one_d(rho, v[1], v[2]);
  x[1] = (x[2] - v[2]) * x[2] / rho;
  x[0] = v[0] - rho;
}

double exp_calc_grad(double *v, double *x, double rho) {
  exp_solve_for_x_with_rho(v, x, rho);
  if (x[1] <= CONE_THRESH_TWO) {
    return x[0];
  }
  return x[0] + x[1] * log(x[1] / x[2]);
}

void exp_get_rho_ub(double *v, double *x, double *ub, double *lb) {
  *lb = 0;
  *ub = 0.125;
  while (exp_calc_grad(v, x, *ub) > 0) {
    *lb = *ub;
    (*ub) *= 2;
  }
}

// int isin_kexp(double r, double s, double t){
//     return ((s * exp(r / s) - t <= CONE_THRESH && s > 0) ||
//       (r <= CONE_THRESH && fabs(s) <= CONE_THRESH && t >= 0));
// }

int isin_kexp(double * z){
    return (((z[1] * exp(z[0] / z[1]) - z[2] <= CONE_THRESH) && (z[1] > 0)) ||
      ((z[0] <= 0) && (fabs(z[1]) <= CONE_THRESH) && (z[2] >= 0)));
}

// int isin_minus_kexp_star(double r, double s, double t){
//     return ((-r < 0 && r * exp(s / r) + EulerConstant * t <= CONE_THRESH) ||
//       (fabs(r) <= CONE_THRESH && -s >= 0 && -t >= 0));
// }

int isin_minus_kexp_star(double * z){
    double r = z[0],s = z[1],t = z[2];
    return (((-r < 0) && (r * exp(s / r) + EulerConstant * t <= CONE_THRESH)) ||
      ((fabs(r) <= CONE_THRESH) && (-s >= 0) && (-t >= 0)));
}

int isin_special_case(double * z){
    return ((z[0] <= CONE_THRESH) && (z[1] <= 0));
}

void exp_cone_projection(double *z) {
  
  int i;
  double ub, lb, g, x[3];
  
  //double r = z[0], s = z[1], t = z[2];
  
  double tol = CONE_TOL;

  /* v in cl(Kexp) */
  //if (isin_kexp(r,s,t)) {
  if (isin_kexp(z)) {
    return;
  }

  /* -v in Kexp^* */
  //if (isin_minus_kexp_star(r, s, t)){
  if (isin_minus_kexp_star(z)){
    memset(z, 0, 3 * sizeof(double));
    return;
  }

  /* special case with analytical solution */
  if (isin_special_case(z)) {
    z[1] = 0.0;
    z[2] = (z[2] > 0.0) ? z[2] : 0.0;
    return;
  }

  /* iterative procedure to find projection, bisects on dual variable: */
  exp_get_rho_ub(z, x, &ub, &lb); /* get starting upper and lower bounds */
  for (i = 0; i < EXP_CONE_MAX_ITERS; ++i) {
    double rho = (ub + lb) / 2;          /* halfway between upper and lower bounds */
    g = exp_calc_grad(z, x, rho); /* calculates gradient wrt dual var */
    if (g > 0) {
      lb = rho;
    } else {
      ub = rho;
    }
    if (ub - lb < tol) {
      break;
    }
  }

  z[0] = x[0];
  z[1] = x[1];
  z[2] = x[2];
  return;
}



int exp_cone_projection_derivative(double *z, 
                                    double *dz, 
                                    double *pi_z){

    // double r = z[0];
    // double s = z[1];
    // double t = z[2];

    // double dr = dz[0];
    // double ds = dz[1];
    // double dt = dz[2];

    // double x = pi_z[0];
    // double y = pi_z[1];
    // double z = pi_z[2];

    //if (isin_kexp(z[0],z[1],z[2])){
    if (isin_kexp(z)) {
        return 1;
    };

    // if ((s > 0) && (s * exp(r / s) < t)){
    //     return;
    // };
        
    // if (((-r < 0) && (r * exp(s / r) == -EulerConstant * t)) || 
    //         ((r == 0) && (-s >= 0) && (-t >= 0))){
    //     dz[0] = 0.0;
    //     dz[1] = 0.0;
    //     dz[2] = 0.0;
    //     return;
    // };

      /* -v in Kexp^* */
      //if (isin_minus_kexp_star(z[0],z[1],z[2])){
      if (isin_minus_kexp_star(z)){
        memset(dz, 0, 3 * sizeof(double));
        return 1;
      }

      /* special case with analytical solution */
      if (isin_special_case(z)) {
        dz[1] = 0.0;
        if (z[2] < 0.0) {dz[2] = 0.0;}
        return 1;
      }

    double jacobian[16]; 

    int success = compute_jacobian_exp_cone(jacobian,
                                          pi_z[2] - z[2],
                                          pi_z[0], pi_z[1], pi_z[2]);

    if (success == 0){
        dz[0] = 0.0;
        dz[1] = 0.0;
        dz[2] = 0.0;
    };

    double old_dz[3];

    old_dz[0] = dz[0];
    old_dz[1] = dz[1];
    old_dz[2] = dz[2];


    // TODO maybe use BLAS
    dz[0] = jacobian[0] * old_dz[0] + jacobian[1] * old_dz[1] + jacobian[2] * old_dz[2];
    dz[1] = jacobian[4] * old_dz[0] + jacobian[5] * old_dz[1] + jacobian[6] * old_dz[2];
    dz[2] = jacobian[8] * old_dz[0] + jacobian[9] * old_dz[1] + jacobian[10] * old_dz[2];

    return success;



    // if (-r < 0 and r * np.exp(s / r) < -np.exp(1) * t):  # or \
    //        # (r == 0 and -s > 0 and -t > 0):
    //     # print('second case')
    //     return np.zeros(3)

    // if r < 0 and s < 0 and t == 0:
    //     # raise NonDifferentiable
    //     return np.zeros(3)

    // # third case
    // if r < 0 and s < 0:
    //     # print('third case')
    //     result = np.zeros(3)
    //     result[0] = dz[0]
    //     result[2] = dz[2] if t > 0 else 0.
    //     # print('result', result)
    //     return result

    // # fourth case
    // #fourth = fourth_case_D(r, s, t, x, y, z, dr, ds, dt)
    // fourth = fourth_case_D_new(r, s, t, x, y, z, dr, ds, dt)
    // # assert not True in np.isnan(fourth)
    // return fourth


}

int compute_jacobian_exp_cone(double *result, double mu_star,
                              double x_star, double y_star, 
                              double z_star){
    /* From BMB'18 appendix C. */

    if (y_star == 0){
        /*Can't compute derivative.*/
        return 0;
    }

    double matrix[16];
    double alpha = x_star / y_star;
    double beta = exp(alpha);
    double gamma = mu_star * beta / y_star;

    matrix[0] = 1 + gamma;
    matrix[1] = - gamma * alpha;
    matrix[2] = 0;
    matrix[3] = beta;

    matrix[4] = matrix[1];
    matrix[5] = 1 + gamma * alpha * alpha;
    matrix[6] = 0;
    matrix[7] = (1 - alpha) * beta;

    matrix[8] = 0;
    matrix[9] = 0;
    matrix[10] = 1;
    matrix[11] = -1;

    matrix[12] = beta;
    matrix[13] = matrix[7];
    matrix[14] = -1;
    matrix[15] = 0;

    return inverse_four_by_four(matrix, result);
}


int inverse_four_by_four(const double m[16], double invOut[16])
{
    double inv[16], det;
    int i;

    inv[0] = m[5]  * m[10] * m[15] - 
             m[5]  * m[11] * m[14] - 
             m[9]  * m[6]  * m[15] + 
             m[9]  * m[7]  * m[14] +
             m[13] * m[6]  * m[11] - 
             m[13] * m[7]  * m[10];

    inv[4] = -m[4]  * m[10] * m[15] + 
              m[4]  * m[11] * m[14] + 
              m[8]  * m[6]  * m[15] - 
              m[8]  * m[7]  * m[14] - 
              m[12] * m[6]  * m[11] + 
              m[12] * m[7]  * m[10];

    inv[8] = m[4]  * m[9] * m[15] - 
             m[4]  * m[11] * m[13] - 
             m[8]  * m[5] * m[15] + 
             m[8]  * m[7] * m[13] + 
             m[12] * m[5] * m[11] - 
             m[12] * m[7] * m[9];

    inv[12] = -m[4]  * m[9] * m[14] + 
               m[4]  * m[10] * m[13] +
               m[8]  * m[5] * m[14] - 
               m[8]  * m[6] * m[13] - 
               m[12] * m[5] * m[10] + 
               m[12] * m[6] * m[9];

    inv[1] = -m[1]  * m[10] * m[15] + 
              m[1]  * m[11] * m[14] + 
              m[9]  * m[2] * m[15] - 
              m[9]  * m[3] * m[14] - 
              m[13] * m[2] * m[11] + 
              m[13] * m[3] * m[10];

    inv[5] = m[0]  * m[10] * m[15] - 
             m[0]  * m[11] * m[14] - 
             m[8]  * m[2] * m[15] + 
             m[8]  * m[3] * m[14] + 
             m[12] * m[2] * m[11] - 
             m[12] * m[3] * m[10];

    inv[9] = -m[0]  * m[9] * m[15] + 
              m[0]  * m[11] * m[13] + 
              m[8]  * m[1] * m[15] - 
              m[8]  * m[3] * m[13] - 
              m[12] * m[1] * m[11] + 
              m[12] * m[3] * m[9];

    inv[13] = m[0]  * m[9] * m[14] - 
              m[0]  * m[10] * m[13] - 
              m[8]  * m[1] * m[14] + 
              m[8]  * m[2] * m[13] + 
              m[12] * m[1] * m[10] - 
              m[12] * m[2] * m[9];

    inv[2] = m[1]  * m[6] * m[15] - 
             m[1]  * m[7] * m[14] - 
             m[5]  * m[2] * m[15] + 
             m[5]  * m[3] * m[14] + 
             m[13] * m[2] * m[7] - 
             m[13] * m[3] * m[6];

    inv[6] = -m[0]  * m[6] * m[15] + 
              m[0]  * m[7] * m[14] + 
              m[4]  * m[2] * m[15] - 
              m[4]  * m[3] * m[14] - 
              m[12] * m[2] * m[7] + 
              m[12] * m[3] * m[6];

    inv[10] = m[0]  * m[5] * m[15] - 
              m[0]  * m[7] * m[13] - 
              m[4]  * m[1] * m[15] + 
              m[4]  * m[3] * m[13] + 
              m[12] * m[1] * m[7] - 
              m[12] * m[3] * m[5];

    inv[14] = -m[0]  * m[5] * m[14] + 
               m[0]  * m[6] * m[13] + 
               m[4]  * m[1] * m[14] - 
               m[4]  * m[2] * m[13] - 
               m[12] * m[1] * m[6] + 
               m[12] * m[2] * m[5];

    inv[3] = -m[1] * m[6] * m[11] + 
              m[1] * m[7] * m[10] + 
              m[5] * m[2] * m[11] - 
              m[5] * m[3] * m[10] - 
              m[9] * m[2] * m[7] + 
              m[9] * m[3] * m[6];

    inv[7] = m[0] * m[6] * m[11] - 
             m[0] * m[7] * m[10] - 
             m[4] * m[2] * m[11] + 
             m[4] * m[3] * m[10] + 
             m[8] * m[2] * m[7] - 
             m[8] * m[3] * m[6];

    inv[11] = -m[0] * m[5] * m[11] + 
               m[0] * m[7] * m[9] + 
               m[4] * m[1] * m[11] - 
               m[4] * m[3] * m[9] - 
               m[8] * m[1] * m[7] + 
               m[8] * m[3] * m[5];

    inv[15] = m[0] * m[5] * m[10] - 
              m[0] * m[6] * m[9] - 
              m[4] * m[1] * m[10] + 
              m[4] * m[2] * m[9] + 
              m[8] * m[1] * m[6] - 
              m[8] * m[2] * m[5];

    det = m[0] * inv[0] + m[1] * inv[4] + m[2] * inv[8] + m[3] * inv[12];

    if (det == 0)
        return 0;

    det = 1.0 / det;

    for (i = 0; i < 16; i++)
        invOut[i] = inv[i] * det;

    return 1;
}

void semidefinite_cone_projection(double *z, double *pi_z, int semidefinite, 
                                  double *eigenvectors, double *eigenvalues){

}


// void embedded_cone_projection(double *z, double *pi_z, int zero, int non_negative, 
//                               int *second_order_dimensions, int second_order_count,
//                               int *semidefinite_dimensions, int semidefinite_count,
//                               double *eigenvectors, double *eigenvalues,
//                               int exponential_primal, int exponential_dual){

//     int cur = 0;

//     zero_cone_projection(pi_z, zero);
//     cur += zero;

//     non_negative_cone_projection(z + cur, pi_z + cur, non_negative);
//     cur += non_negative;

//     for (i = 0; i < second_order_count; i++){
//         second_order_cone_projection(z + cur, pi_z + cur, second_order_dimensions[i]);
//         cur += second_order_dimensions[i];
//     };

//     for (i = 0; i < semidefinite_count; i++){
//         semidefinite_cone_projection(z + cur, pi_z + cur, semidefinite_dimensions[i], 
//                                     );
//         cur += semidefinite_dimensions[i];
//     };

//     non_negative_cone_projection(z + cur, pi_z + cur, 1);
// };


