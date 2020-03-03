/*
*  Cone Program Refinement
*
*  Copyright (C) 2020 Enzo Busseti
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


#include <math.h>
#include "mini_cblas.h"
#include <string.h>


/* sqrt(a**2 + b**2) that limits overflow */
static double d2norm(const double a, const double b){
    
    double scale, scaled_a, scaled_b;

    scale = fabs(a) + fabs(b);

    if (scale == 0.)
        return 0.;

    scaled_a = a / scale;
    scaled_b = b / scale;

    return scale * sqrt(pow(scaled_a, 2.) + pow(scaled_b,2));

}


void truncated_lsqr(const int m, 
    const int n,
    void (*aprod)(const int mode, const int m, const int n, 
                  double * x, double * y, void *UsrWrk),
    const double * b, /*m-vector*/
    const int max_iter,
    double * x, /*result n-vector*/
    double * u, /*internal m-vector*/
    double * v, /*internal n-vector*/
    double * w, /*internal n-vector*/
    void *UsrWrk /*workspace for aprod*/
    ){

    double alpha, beta, phi_bar, rho,rho_bar, c, s,theta,phi;
    int i;

    /*For safety */
    memset(x, 0, sizeof(double) * n);
    memset(v, 0, sizeof(double) * n);

    /*beta = norm2(u) */
    beta = cblas_dnrm2(m, b, 1);
    if (beta == 0) 
        return;
    /*u = b/beta */
    memcpy(u, (const double*) b, sizeof(double) * m);
    cblas_dscal(m, 1./beta, u, 1);


    /* v = rmatvec(u) */
    aprod ( 2, m, n, v, u, UsrWrk );
    /*alpha = norm2(v) */
    alpha = cblas_dnrm2(n, v, 1);
    if (alpha == 0)
        return;
    /*v /= alpha */
    cblas_dscal(n, 1./alpha, v, 1);

    /*w = v*/
    memcpy(w, (const double*) v, sizeof(double) * n);

    phi_bar = beta;
    rho_bar = alpha;

    for (i=0; i < max_iter; i++){

        /* continue the bidiagonalization */

        /* u /= (-alpha)*/
        cblas_dscal(m, (- alpha), u, 1 );

        /* u += matvec(v) */
        aprod ( 1, m, n, v, u, UsrWrk );

        /*beta = norm2(u) */
        beta = cblas_dnrm2(m, u, 1);
        if (beta == 0) 
            return;
        /*u /= beta */
        cblas_dscal(m, 1./beta, u, 1);

        /* v /= (-beta)*/
        cblas_dscal (n, (- beta), v, 1 );

        /* v += rmatvec(u) */
        aprod ( 2, m, n, v, u, UsrWrk );

        /*alpha = norm2(v) */
        alpha = cblas_dnrm2(n, v, 1);
        if (alpha == 0)
             return;

        /*v /= alpha */
        cblas_dscal(n, 1./alpha, v, 1);


        rho = d2norm(rho_bar, beta);

        c = rho_bar / rho;
        s = beta / rho;

        theta = s * alpha;
        rho_bar = -c * alpha;
        phi = c * phi_bar;
        phi_bar = s * phi_bar;

        /* update x and w */

        /* x = x + (phi / rho) * w */
        cblas_daxpy(n, (phi / rho), 
            (const double * )w, 1, x, 1);

        /* w *= - (theta / rho) */
        cblas_dscal(n, -(theta / rho), w, 1);
        
        /* w += v*/
        cblas_daxpy(n, 1., (const double * )v, 1, w, 1);


    }

    return;

}
