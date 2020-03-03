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

/*
* Implementation of LSQR following the algorithm description
* in "LSQR: An Algorithm for Sparse Linear Equations and Sparse
* Least Squares", by C. Paige and M. Saunders, 1982.
*
* The only termination criteria used is the maximum number
* of iterations max_iter.


*     The matrix A is intended to be large and sparse.  It is accessed
*     by means of subroutine calls of the form
*
*                call aprod ( mode, m, n, x, y, leniw, lenrw, iw, rw )
*
*     which must perform the following functions:
*
*                If mode = 1, compute  y = y + A*x.
*                If mode = 2, compute  x = x + A(transpose)*y.
*
*     The vectors x and y are input parameters in both cases.
*     If  mode = 1,  y should be altered without changing x.
*     If  mode = 2,  x should be altered without changing y.
*     The parameters leniw, lenrw, iw, rw may be used for workspace
*     as described below.
*/


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
    );



