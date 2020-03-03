#include "test.h"
#include "cones.h"

static const char * test_embedded_cone_projection(){

    int lensol = LENSOL;
    int lenzero = LENZERO;
    int lennonneg = LENNONEG;
    
    double z[N], pi_z[N];
    int i,j,k;

    for (j=0; j<NUM_CONES_TESTS; j++){
        random_uniform_vector(N, z, -1, 1,j);
        embedded_cone_projection(z, pi_z, lensol, lenzero, lennonneg);
       
       if (DEBUG_PRINT){
        printf("\nTesting cone projection\n"); 
        for (i= 0; i<N;i++){
            printf("z[%d] = %f, pi_z[%d] = %f\n", i, z[i], i, pi_z[i]);
         }
     }


        for (k = 0; k < lensol + lenzero; k++)
            mu_assert("real cone projection error", pi_z[k] == z[k]);
        
        for (k = lensol + lenzero; k < lensol +lenzero + lennonneg; k++)
            mu_assert("non-neg cone projection error", pi_z[k] >= 0.);
        
        mu_assert("error, pi_z[-1] < 0", pi_z[N-1] >= 0);
        }
     return 0;
 }


 static const char * test_embedded_cone_projection_derivative() {
    double z[N], pi_z[N], dz[N], z_p_dz[N], dpi_z[N], pi_z_p_dz[N];
    int i,j, k;
    int equal;


    for (j=0; j<NUM_CONES_TESTS; j++){
        random_uniform_vector(N, z, -1, 1, j*1234);
        random_uniform_vector(N, dz, -1E-8, 1E-8, j*5678);
        embedded_cone_projection(z, pi_z, LENSOL, LENZERO, LENNONEG);

        for (k = 0; k < NUM_BACKTRACKS; k++){

        if (DEBUG_PRINT) printf("\nscaling dz by (0.9)^%d\n",k);

        for (i= 0; i<N;i++) z_p_dz[i] = z[i] + dz[i];

        embedded_cone_projection(z_p_dz, pi_z_p_dz, LENSOL, LENZERO, LENNONEG);

        embedded_cone_projection_derivative(z, pi_z, dz, dpi_z, LENSOL, LENZERO, LENNONEG);

        if (DEBUG_PRINT){
        printf("\nTesting cone projection derivative\n");
        for (i= 0; i<N;i++){
           printf("z[%d] = %.2f, pi_z[%d] = %.2f, dz[%d] = %.2f, dpi_z[%d] = %.2f, pi_z_p_dz[%d] = %.2f\n", 
               i, z[i], i, pi_z[i], i, dz[i], i, dpi_z[i], i, pi_z_p_dz[i]);
        }
    }

        /* pi_z + dpi_z == pi_z_p_dz*/
        equal = 0;
        for (i = 0; i <N; i++){
            if (DEBUG_PRINT)
                    printf("error[%d] = %e\n", i,
                        pi_z[i] + dpi_z[i] - pi_z_p_dz[i]);
            if (fabs(pi_z[i] + dpi_z[i] - pi_z_p_dz[i])>1E-15) {
                equal = -1;
                break;} 
        }
        if (equal == 0) break;
        

        for (i= 0; i<N;i++) dz[i] *= (0.9);
}

        mu_assert("error, pi_z + dpi_z != pi_z_p_dz", !equal);

        }
     return 0;
 }
