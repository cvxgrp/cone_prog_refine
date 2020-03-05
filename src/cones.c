/*
*  CPR - Cone Program Solution Refinement
*
*  Copyright (C) 2019-2020, Enzo Busseti, Walaa Moursi, and Stephen Boyd
*
* Licensed under the Apache License, Version 2.0 (the "License");
* you may not use this file except in compliance with the License.
* You may obtain a copy of the License at
* 
*    http://www.apache.org/licenses/LICENSE-2.0
* 
* Unless required by applicable law or agreed to in writing, software
* distributed under the License is distributed on an "AS IS" BASIS,
* WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
* See the License for the specific language governing permissions and
* limitations under the License.
*/

#include "cones.h"
#include "math.h"
#include "string.h"
#include "stdbool.h"
#include "mini_cblas.h"

int embedded_cone_projection(
    const double * z, 
    double * pi_z,
    const vecsize size_solution,
    const vecsize size_zero, 
    const vecsize size_non_neg,
    const vecsize num_second_order,
    const vecsize * sizes_second_order
    /*const vecsize num_exp_pri,
    const vecsize num_exp_dua*/
    ){

    int i, counter;

    /*Zero cone.*/
    counter = size_solution + size_zero;
    memcpy(pi_z, z, sizeof(double) * (counter));

    /*Non-negative cone.*/
    for (i = counter; i < counter + size_non_neg; i++){
        pi_z[i] = z[i] <= 0 ? 0 : z[i];
    }
    counter += size_non_neg;

    /* Second order cones. */
    for (i = 0; i < num_second_order; i++){  
        second_order_cone_projection(
            z + counter, 
            pi_z + counter,
            sizes_second_order[i]);
        counter += sizes_second_order[i];
    };

    /*Last element of the embedded cone.*/
    pi_z[counter] = z[counter] <= 0 ? 0 : z[counter];

    return 0;

}

/*TODO return -1 if non-differentiable.*/
/*TODO add transpose.*/
int embedded_cone_projection_derivative(
    const double * z, 
    const double * pi_z,
    const double * dz,
    double * dpi_z,
    const vecsize size_solution,
    const vecsize size_zero, 
    const vecsize size_non_neg,
    const vecsize num_second_order,
    const vecsize * sizes_second_order
    /*
    const vecsize num_exp_pri,
    const vecsize num_exp_dua*/
    ){

    int i, counter;

    /*Zero cone.*/
    counter = size_solution + size_zero;
    memcpy(dpi_z, dz, sizeof(double) * (counter));

    /*Non-negative cone.*/
    for (i = counter; i < counter + size_non_neg; i++){
        dpi_z[i] = z[i] <= 0. ? 0. : dz[i];
    }
    counter += size_non_neg;

    /*Second order cones.*/

    /*Last element */
    dpi_z[counter] = z[counter] <= 0. ? 0. : dz[counter];

    return 0;

}


void second_order_cone_projection(
    const double *z, 
    double * pi_z,
    const vecsize size){

    double norm_x, rho, mult;
    int i;

    norm_x = cblas_dnrm2(size - 1, z + 1, 1);

    if (norm_x <= z[0]){
        memcpy(pi_z, z, sizeof(double) * size);
        return;
    }

    if (norm_x <= -z[0]){
        memset(pi_z, 0, sizeof(double) * size);
        return;
    }

    rho = z[0];

    pi_z[0] = (norm_x + rho) / 2.;

    mult = pi_z[0]/norm_x;

    for (i = 1; i < size; i++){
        pi_z[i] = z[i] * mult;
    } 
}



/*
void zero_cone_projection_derivative(const double *z, double *dz, 
                                     const vecsize size){
    memset(dz, 0, sizeof(double) * size);
}


void non_negative_cone_projection_derivative(const double *z, double *dz, const vecsize size){
    int i;

    for (i = 0; i < size; i++){
        if (z[i] < 0) {
            dz[i] = 0;
        };
    }
}
*/




void second_order_cone_projection_derivative(const double *z, 
                                             double *dz, 
                                             const double *pi_z,
                                             const vecsize size){


    double norm_x, dot_val, old_dzzero, first_coefficient, second_coefficient;

    norm_x = cblas_dnrm2(size - 1, z + 1, 1);

    if (norm_x <= z[0]){return;}
    
    if (norm_x <= -z[0]){
        memset(dz, 0, sizeof(double) * size);
        return;
    }

    dot_val = cblas_ddot(size - 1, dz + 1, 1, z + 1, 1);

    old_dzzero = dz[0];
    
    dz[0] = (dz[0] * norm_x + dot_val) / (2 * norm_x);

    first_coefficient = (z[0] + norm_x) / (2. * norm_x);
    second_coefficient = (old_dzzero - z[0] * dot_val / (norm_x*norm_x)) / (2. * norm_x);

    cblas_dscal(size - 1, first_coefficient, dz + 1, 1);
    cblas_daxpy(size - 1, second_coefficient, z + 1, 1, dz + 1, 1);

    return;

}


#define CONE_TOL (1e-16)
#define CONE_THRESH (1e-16)
#define CONE_THRESH_TWO (1e-16)
#define EXP_CONE_MAX_ITERS (200)

const double EulerConstant = 2.718281828459045; 

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


int isin_kexp(double * z){
    return (((z[1] * exp(z[0] / z[1]) - z[2] <= CONE_THRESH) && (z[1] > 0)) ||
      ((z[0] <= 0) && (fabs(z[1]) <= CONE_THRESH) && (z[2] >= 0)));
}


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
  
  
  double tol = CONE_TOL;

  if (isin_kexp(z)) {
    return;
  }

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

    double jacobian[16], old_dz[3];
    int success;

    if (isin_kexp(z)) {
        return 1;
    };

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

    success = compute_jacobian_exp_cone(jacobian,
                                          pi_z[2] - z[2],
                                          pi_z[0], pi_z[1], pi_z[2]);

    if (success == 0){
        dz[0] = 0.0;
        dz[1] = 0.0;
        dz[2] = 0.0;
    };

    old_dz[0] = dz[0];
    old_dz[1] = dz[1];
    old_dz[2] = dz[2];


    /* TODO maybe use BLAS*/
    dz[0] = jacobian[0] * old_dz[0] + jacobian[1] * old_dz[1] + jacobian[2] * old_dz[2];
    dz[1] = jacobian[4] * old_dz[0] + jacobian[5] * old_dz[1] + jacobian[6] * old_dz[2];
    dz[2] = jacobian[8] * old_dz[0] + jacobian[9] * old_dz[1] + jacobian[10] * old_dz[2];

    return success;

}

int compute_jacobian_exp_cone(double *result, double mu_star,
                              double x_star, double y_star, 
                              double z_star){
    /* From BMB'18 appendix C. */

    double matrix[16], alpha, beta, gamma;

    alpha = x_star / y_star;
    beta = exp(alpha);
    gamma = mu_star * beta / y_star;

    if (y_star == 0){
        /*Can't compute derivative.*/
        return 0;
    }

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



/*****************
Semi-definite cone
*****************/


vecsize sizevec2sizemat(vecsize n){
    vecsize m;

    m = floor(sqrt(2 * n));
    if (n != (m * (m + 1) / 2.0)){
        return -1;
    }
    return m;
}


vecsize sizemat2sizevec(vecsize m){
    return ((m * (m + 1) / 2));
}


int mat2vec(double * Z, vecsize n, double * z, vecsize m){
    /*Upper tri row stacked, off diagonal multiplied by sqrt(2).*/

    vecsize cur = 0;
    int i; 
    const double sqrt_two = 1.4142135623730951;
    
    /*Scale down the diagonal.*/
    cblas_dscal(n, 1./sqrt_two, Z, n+1);

    for (i = 0; i<n; i++){  
        memcpy(z + cur, Z + (i * n) + i, sizeof(double) * (n - i));
        cur += (n - i);
    };

    /*Scale by sqrt(2).*/
    cblas_dscal(sizemat2sizevec(n), sqrt_two, z, 1);

    return 0;
}


int vec2mat(double * z, vecsize m, double * Z, vecsize n){

    vecsize cur = 0;
    int i, j;
    const double sqrt_two = sqrt(2);

    for (i = 0; i<n; i++){  
        Z[n * i + i] = z[cur];
        cur ++;
        for (j = i+1; j < n; j++){ 
            Z[n * i + j] = z[cur] / sqrt_two;
            Z[n * j + i] = Z[n * i + j];
            cur ++;
        };
    };

    if (cur != m){
        return -1;
    };

    return 0;

}


void semidefinite_cone_projection(double *z, 
                                  const vecsize semidefinite, 
                                  double *eigenvectors, 
                                  double *eigenvalues){
/*
    const vecsize matsize = sizevec2sizemat(semidefinite);



    dsyev_("Vectors", "All", "Lower", semidefinite,
        __CLPK_doublereal *__a, __CLPK_integer *__lda, __CLPK_doublereal *__vl,
        __CLPK_doublereal *__vu, __CLPK_integer *__il, __CLPK_integer *__iu,
        __CLPK_doublereal *__abstol, __CLPK_integer *__m,
        __CLPK_doublereal *__w, __CLPK_doublereal *__z__, __CLPK_integer *__ldz,
        __CLPK_integer *__isuppz, __CLPK_doublereal *__work,
        __CLPK_integer *__lwork, __CLPK_integer *__iwork,
        __CLPK_integer *__liwork,
        __CLPK_integer *__info)
        */

}


/*
void embedded_cone_projection(double *z, double *pi_z, int zero, int non_negative, 
                              int *second_order_dimensions, int second_order_count,
                              int *semidefinite_dimensions, int semidefinite_count,
                              double *eigenvectors, double *eigenvalues,
                              int exponential_primal, int exponential_dual){

    int cur = 0;

    zero_cone_projection(pi_z, zero);
    cur += zero;

    non_negative_cone_projection(z + cur, pi_z + cur, non_negative);
    cur += non_negative;

    for (i = 0; i < second_order_count; i++){
        second_order_cone_projection(z + cur, pi_z + cur, second_order_dimensions[i]);
        cur += second_order_dimensions[i];
    };

    for (i = 0; i < semidefinite_count; i++){
        semidefinite_cone_projection(z + cur, pi_z + cur, semidefinite_dimensions[i], 
                                    );
        cur += semidefinite_dimensions[i];
    };

    non_negative_cone_projection(z + cur, pi_z + cur, 1);
};

*/
