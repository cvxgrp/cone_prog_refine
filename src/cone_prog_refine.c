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

#include "cone_prog_refine.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "problem.h"
#include "mini_cblas.h"
#include "lsqr.h"
#include <math.h>

int cone_prog_refine_alloc(
    const int m, 
    const int n,
    double ** norm_res,
    double ** pi_z,
    double ** internal,  /*Used by DN(z)*/
    double ** internal2,  /*Used by DN(z)*/
    double ** u, /*Used by LSQR*/
    double ** v, /*Used by LSQR*/
    double ** w, /*Used by LSQR*/
    double ** delta /*Used by LSQR*/
    ){

    void * allocated_memory;

    allocated_memory = malloc(8*(n+m+1) * sizeof(double));

    if (allocated_memory == NULL) /*Allocation failed*/
        return -1;
    
    *norm_res = (double *) allocated_memory;
    *pi_z = ((double *) allocated_memory) + (n + m + 1);
    /*Used by DN(z)*/
    *internal = ((double *) allocated_memory) + 2*(n + m + 1);
    *internal2 = ((double *) allocated_memory) + 3*(n + m + 1);
    /*Used by LSQR*/
    *u = ((double *) allocated_memory) + 4*(n + m + 1);
    *v = ((double *) allocated_memory) + 5*(n + m + 1);
    *w = ((double *) allocated_memory) + 6*(n + m + 1);
    /*LSQR result*/
    *delta = ((double *) allocated_memory) + 7*(n + m + 1);

    return 0;

}




int cone_prog_refine(
    const int m, /*number of rows of A*/
    const int n, /*number of columns of A*/
    const int size_zero, /*size of zero cone*/
    const int size_nonneg, /*size of non-negative cone*/
    const int num_sec_ord, /*number of second order cones*/
    const int *sizes_sec_ord, /*sizes of second order cones*/
    const int num_exp_pri, /*number of exponential primal cones*/
    const int num_exp_dua, /*number of exponential dual cones*/
    const int * A_col_pointers, /*pointers to columns of A, in CSC format*/
    const int * A_row_indeces, /*indeces of rows of A, in CSC format*/
    const double * A_data, /*elements of A, in CSC format*/
    const double * b, /*m-vector*/
    const double * c, /*n-vector*/
    double * z, /* (m+n+1)-vector, 
                    approximate primal-dual embedded solution,
                    will be overwritten by refined solution*/
    const int num_lsqr_iters, /*number of lsqr iterations*/
    const double lambda,
    const int num_iters, /*number of refine iterations*/
    const int print_info /*print informations on convergence*/
    ){

    int nondiff, i, k;
    double * norm_res;
    double * pi_z;
    double * delta;
    double *u, *v, *w;
    int alloc_success;
    int j;

    /*LSQR result*/
    int istop_out, itn_out;
    double anorm_out, acond_out, rnorm_out, arnorm_out, xnorm_out;


    double old_normres, new_normres;

    lsqr_workspace workspace; 

    /*Assign constants to workspace used by LSQR.*/
    workspace.m = m;
    workspace.n = n;
    workspace.size_zero = size_zero;
    workspace.size_nonneg = size_nonneg;
    workspace.num_sec_ord = num_sec_ord;
    workspace.sizes_sec_ord = sizes_sec_ord;
    workspace.num_exp_pri = num_exp_pri;
    workspace.num_exp_dua = num_exp_dua;
    workspace.A_col_pointers = A_col_pointers;
    workspace.A_row_indeces = A_row_indeces;
    workspace.A_data = A_data;
    workspace.b = b;
    workspace.c = c;

    alloc_success = cone_prog_refine_alloc(
                    m, 
                    n,
                    &norm_res,
                    &pi_z,
                    &workspace.internal,  /*Used by DN(z)*/
                    &workspace.internal2,  /*Used by DN(z)*/
                    &u, /*Used by LSQR*/
                    &v, /*Used by LSQR*/
                    &w, /*Used by LSQR*/
                    &delta /*Used by LSQR*/
                    );

    /*Compute normalized residual*/
    nondiff = projection_and_normalized_residual(
        m, n, size_zero, size_nonneg, num_sec_ord, sizes_sec_ord,
        num_exp_pri, num_exp_dua, A_col_pointers, A_row_indeces,
        A_data, b, c, norm_res, pi_z, z);

    old_normres = cblas_dnrm2(n+m+1, norm_res, 1);

    if (print_info)
        printf("Initial ||N(z)|| = %e\n\n", old_normres);


    for (i = 0; i < num_iters; i++){

        /*cblas_dscal(m+n+1, 1./fabs(z[m+n]), z, 1);*/

        workspace.z = z;
        workspace.pi_z = pi_z; /*Used by cone derivatives.*/
        workspace.norm_res_z = norm_res; /*Used by second term of derivative*/

        if (print_info>3) printf("\nIteration %d\n", i);
        
    /*u = N(z)*/
    memcpy(u, norm_res, sizeof(double)*(m+n+1));

    lsqr(m+n+1,
      m+n+1,
      normalized_residual_aprod,
      sqrt(lambda), /*damp*/
      (void *)&workspace,
      u,    /* len = m */
      v,    /* len = n */
      w,    /* len = n */
      delta,    /* len = n */
      NULL,   /* len = * */
      0., /*atol*/
      0., /*btol*/
      0., /*conlim*/
      num_lsqr_iters,
      (print_info>2)? stdout:NULL,
      /* The remaining variables are output only. */
      &istop_out,
      &itn_out,
      &anorm_out,
      &acond_out,
      &rnorm_out,
      &arnorm_out,
      &xnorm_out
     );

    if (print_info>1)
        printf("\nanorm_out = %.2e, acond_out = %.2e, rnorm_out = %.2e, \narnorm_out = %.2e, xnorm_out = %.2e\n\n",
        anorm_out, acond_out, rnorm_out, arnorm_out, xnorm_out );
            

    if (print_info>3){
        printf("\n");
            for (j = 0; j < m+n+1; j++)
                printf("delta[%d] = %e\n", j, delta[j]);
            
        printf("\n");
    }

        cblas_daxpy(m+n+1, -1., delta, 1, z, 1);

        for (k = 0; k < MAX_CONE_PROG_REFINE_BACKTRACKS; k++){

            if (print_info>2) printf("Backtrack %d\n", k);

            if (print_info>3) 
                {printf("\n");
            for (j = 0; j < m+n+1; j++)
                printf("z[%d] = %e\n", j, z[j]);
            
            printf("\n");
        }

            nondiff = projection_and_normalized_residual(
            m, n, size_zero, size_nonneg, num_sec_ord, sizes_sec_ord,
            num_exp_pri, num_exp_dua, A_col_pointers, A_row_indeces,
            A_data, b, c, norm_res, pi_z, z);

            new_normres = cblas_dnrm2(n+m+1, norm_res, 1);

            if (print_info>2)
                printf("new ||N(z)|| = %e\n\n", new_normres);

            if (new_normres < old_normres){
                old_normres = new_normres;
                if (print_info)
                    printf("It. %d, %d lsqr_its, %d btrs, ||N(z)|| = %.2e, %s (z[-1] = %.2e)\n",
                        i, itn_out, k, new_normres, z[m+n]>0?"SOL":"CERT", z[m+n]);
                break;
                /*TODO It's not exiting it fails to refine.*/
            }

            if (print_info>2) 
                printf("removing %e times delta\n", pow(.5, k+1) );
            cblas_daxpy(m+n+1, pow(.5, k+1), delta, 1, z, 1);

    }

    if (old_normres < 1E-14){
        break;
    }

}



    return 0;
}


