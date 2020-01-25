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


double norm(double *x, int size){
    
    double norm_x = 0.;
    for (int i = 0; i < size; i++){
        norm_x += x[i] * x[i];
    }
    return sqrt(norm_x);
}

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

void vecalgsum(double *x, double *y, 
                 double alpha, double beta,
                 int size){
    for (int i = 0; i < size; i++){
        x[i] = alpha * x[i] + beta * y[i];
    }
}

double dot(double *x, double *y, int size){
    double result = 0.;
    for (int i = 0; i < size; i++){
        result += x[i] * y[i];
    }
    return result;
}



void zero_cone_projection(double *z, int64_t size){
    memset(z, 0, sizeof(double) * size);
}

void zero_cone_projection_derivative(double *z, double *x, 
                                     int64_t size){
    memset(x, 0, sizeof(double) * size);
}

void non_negative_cone_projection(double *z, int64_t size){
    for (int i = 0; i < size; i++){
        if (z[i] < 0) {
            z[i] = 0;
        };
    }
}

void non_negative_cone_projection_derivative(double *z, double *x, int64_t size){
    for (int i = 0; i < size; i++){
        if (z[i] < 0) {
            x[i] = 0;
        };
    }
}


void second_order_cone_projection(double *z, int64_t size){

    double norm_x = norm(z + 1, size - 1);

    // for (int i = 1; i < size; i++){
    //     norm_x += z[i] * z[i];
    // }
    // norm_x = sqrt(norm_x);

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

void second_order_cone_projection_derivative(double *z, 
                                             double *dz, 
                                             double *pi_z,
                                             int64_t size){

    // point at which we derive
    // double t = z[0];
    // x = z[1:]

    // projection of point
    // s = pi_z[0]
    // y = pi_z[1:]

    // logic for simple cases
    //norm = np.linalg.norm(z[1:])

    double norm_x = norm(z + 1, size - 1);

    if (norm_x <= z[0]){return;}
    
    if (norm_x <= -z[0]){
        memset(dz, 0, sizeof(double) * size);
        return;
    }

    // big case
    double dot_val = dot(dz + 1, z + 1, size - 1);
    double old_dzzero = dz[0];
    
    dz[0] = (dz[0] * norm_x + dot_val) / (2 * norm_x);

    vecalgsum(dz + 1, 
              z + 1, 
              (z[0] + norm_x) / (2. * norm_x), 
              (old_dzzero - z[0] * dot_val / (norm_x*norm_x)) / (2. * norm_x),
              size - 1);

    return;

}


#define CONE_TOL (1e-14)
#define CONE_THRESH (1e-12)
#define EXP_CONE_MAX_ITERS (100)

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
  if (x[1] <= 1e-12) {
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


void exp_cone_projection(double *z) {
  
  int i;
  double ub, lb, g, x[3];
  
  double r = z[0], s = z[1], t = z[2];
  
  double tol = CONE_TOL;

  /* v in cl(Kexp) */
  if ((s * exp(r / s) - t <= CONE_THRESH && s > 0) ||
      (r <= 0 && fabs(s) <= CONE_THRESH && t >= 0)) {
    return;
  }

  /* -v in Kexp^* */
  if ((-r < 0 && r * exp(s / r) + EulerConstant * t <= CONE_THRESH) ||
      (fabs(r) <= CONE_THRESH && -s >= 0 && -t >= 0)) {
    memset(z, 0, 3 * sizeof(double));
    return;
  }

  /* special case with analytical solution */
  if (r < 0 && s < 0) {
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


