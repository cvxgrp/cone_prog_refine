#ifndef CONES_H
#define CONES_H

/*
*  Cone Program Refinement - Copyright (C) 2020 Enzo Busseti
*
* Licensed under the Apache License, Version 2.0 (the "License");
* you may not use this file except in compliance with the License.
* You may obtain a copy of the License at
*
*     http://www.apache.org/licenses/LICENSE-2.0
*
* Unless required by applicable law or agreed to in writing, software
* distributed under the License is distributed on an "AS IS" BASIS,
* WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
* See the License for the specific language governing permissions and
* limitations under the License.
*/


#include <stdint.h>
#include <problem.h>


/*temporarily for interoperability with Python */
typedef int64_t vecsize;


void second_order_cone_projection(
    const double *z, double * pi_z, const int size);


int second_order_cone_projection_derivative(const int size,
                                             const double *z, 
                                             const double *pi_z,
                                             const double *dz,
                                             double *dpi_z
                                             );

/***************
Exponential cone
***************/

double exp_newton_one_d(double rho, double y_hat, double z_hat);
void exp_solve_for_x_with_rho(double *v, double *x, double rho);
double exp_calc_grad(double *v, double *x, double rho);
void exp_get_rho_ub(double *v, double *x, double *ub, double *lb);

int isin_kexp(double * z);
int isin_minus_kexp_star(double * z);
int isin_special_case(double * z);

int inverse_four_by_four(const double m[16], double invOut[16]);
int compute_jacobian_exp_cone(double *result, double mu_star,
                              double x_star, double y_star, 
                              double z_star);

void exp_cone_projection(double *z, double *pi_z);
int exp_cone_projection_derivative(double *z, 
                                    double *dz,
                                    double *dpi_z, 
                                    double *pi_z);

/*****************
Semi-definite cone
*****************/

vecsize sizevec2sizemat(vecsize n);
vecsize sizemat2sizevec(vecsize m);
int mat2vec(double * Z, vecsize n, double * z, vecsize m);
int vec2mat(double * z, vecsize m, double * Z, vecsize n);


void semidefinite_cone_projection(double *z, 
                                  const vecsize semidefinite, 
                                  double *eigenvectors, 
                                  double *eigenvalues);

/************
Embedded cone
************/

/* workspace->pi_z = Pi workspace->z */
int embedded_cone_projection(cone_prog_refine_workspace * workspace);


/* 
if mode==1,  dpi_z = DPi dz + dpi_z
if mode==2,  dpi_z = DPi^T dz + dpi_z
*/
int embedded_cone_projection_derivative(
    cone_prog_refine_workspace * workspace,
    const double * dz,
    double * dpi_z,
    const int mode);

#endif
