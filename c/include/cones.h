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

//void embedded_cone_projection(double *z, double *pi_z, int zero, int nonneg, int* second_order);

#include <stdint.h>
#include <stdbool.h>

double norm(double *x, int size);

void vecsum(double *x, double *y, int size);

void vecdiff(double *x, double *y, int size);

void vecalgsum(double *x, double *y, 
                 double alpha, double beta,
                 int size);

double dot(double *x, double *y, int size);



void zero_cone_projection(double *z, int64_t size);

void zero_cone_projection_derivative(double *z, double *x, 
                                     int64_t size);

void non_negative_cone_projection(double *z, int64_t size);

void non_negative_cone_projection_derivative(double *z, double *x, 
                                             int64_t size);

void second_order_cone_projection(double *z, int64_t size);

void second_order_cone_projection_derivative(double *z, 
                                             double *dz, 
                                             double *pi_z,
                                             int64_t size);

double exp_newton_one_d(double rho, double y_hat, double z_hat);
void exp_solve_for_x_with_rho(double *v, double *x, double rho);
double exp_calc_grad(double *v, double *x, double rho);
void exp_get_rho_ub(double *v, double *x, double *ub, double *lb);
//int isin_kexp(double r, double s, double t);
int isin_kexp(double * z);

//int isin_minus_kexp_star(double r, double s, double t);
int isin_minus_kexp_star(double * z);
int isin_special_case(double * z);

int inverse_four_by_four(const double m[16], double invOut[16]);
int compute_jacobian_exp_cone(double *result, double mu_star,
                              double x_star, double y_star, 
                              double z_star);


void exp_cone_projection(double *z);

int exp_cone_projection_derivative(double *z, 
                                    double *dz, 
                                    double *pi_z);

void semidefinite_cone_projection(double *z, double *pi_z, int semidefinite, 
                                  double *eigenvectors, double *eigenvalues);
