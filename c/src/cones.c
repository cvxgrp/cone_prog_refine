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


