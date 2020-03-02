#include "test.h"
 
#include "test_linalg.c"
#include "test_problem.c"
#include "test_cones.c"

int tests_run = 0;
 
 static const char * all_tests() {
     mu_run_test(test_embedded_cone_projection);
     mu_run_test(test_embedded_cone_projection_derivative);
     mu_run_test(test_csc_matvec);
     mu_run_test(test_csr_matvec);
     mu_run_test(test_Q_matvec);
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
