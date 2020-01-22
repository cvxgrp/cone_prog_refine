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
#include <string.h>


void zero_cone_projection(double *z, int64_t size){
    for (int i = 0; i < size; i++){
        z[i] = 0.;
   };
}

void zero_cone_projection_derivative(double *z, double *x, 
                                     int64_t size){
    for (int i = 0; i < size; i++){
        x[i] = 0.;
   };
}

void non_negative_cone_projection(double *z, int64_t size){
    for (int i = 0; i < size; i++){
        if (z[i] < 0) {
            z[i] = 0;
        };
    }
}

void non_negative_cone_projection_derivative(double *z, 
    double *x, int64_t size){

    for (int i = 0; i < size; i++){
        if (z[i] < 0) {
            x[i] = 0;
        };
    }
}

void second_order_cone_projection(double *z, int second_order){

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


