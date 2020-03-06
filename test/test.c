#include "test.h"
 
#include "test_linalg.c"
#include "test_problem.c"
#include "test_cones.c"
/*#include "test_truncated_lsqr.c"*/
#include "test_lsqr.c"
#include "test_cone_prog_refine.c"



int tests_run = 0;
 
 static const char * all_tests() {
     mu_run_test(test_embedded_cone_projection);
     mu_run_test(test_csc_matvec);
     mu_run_test(test_csr_matvec);
     mu_run_test(test_Q_matvec);
     mu_run_test(test_normalized_residual);
     mu_run_test(test_normalized_residual_matvec);
      mu_run_test(test_Q_vecmat);
    mu_run_test(test_truncated_lsqr1);
    mu_run_test(test_aprod);
    mu_run_test(test_lsqr);
    mu_run_test(test_second_order_cone);
        mu_run_test(test_embedded_cone_projection_derivative);

    mu_run_test(test_cone_prog_refine);
    mu_run_test(test_exp_cone_proj);
    mu_run_test(test_normalized_residual_vecmat);



     return 0;
 }
 
 int main(int argc, char **argv) {
     const char *result = all_tests();
     if (result != 0) {
         printf("%s\n", result);
     }
     else {
         printf("ALL TESTS PASSED\n");
     }
     printf("Tests run: %d\n", tests_run);
 
     return result != 0;
 }
