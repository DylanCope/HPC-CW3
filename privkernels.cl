#pragma OPENCL EXTENSION cl_khr_fp64 : enable

#define NSPEEDS 9

kernel void accelerate_flow(global float* cells,
                            global int* obstacles,
                            int nx, int ny,
                            float density, float accel)
{
  /* compute weighting factors */
  float w1 = density * accel / 9.0;
  float w2 = density * accel / 36.0;

  /* modify the 2nd row of the grid */
  int ii = ny - 2;
  const int N = nx * ny;
  /* get column index */
  int jj = get_global_id(0);

  /* if the cell is not occupied and
  ** we don't send a negative density */
  if (!obstacles[ii * nx + jj]
      && (cells[3*N + ii * nx + jj] - w1) > 0.0
      && (cells[6*N + ii * nx + jj] - w2) > 0.0
      && (cells[7*N + ii * nx + jj] - w2) > 0.0)
  {
    /* increase 'east-side' densities */
    cells[1*N + ii * nx + jj] += w1;
    cells[5*N + ii * nx + jj] += w2;
    cells[8*N + ii * nx + jj] += w2;
    /* decrease 'west-side' densities */
    cells[3*N + ii * nx + jj] -= w1;
    cells[6*N + ii * nx + jj] -= w2;
    cells[7*N + ii * nx + jj] -= w2;
  }
}

kernel void propagate(global float* cells,
                      global float* tmp_cells,
                      global int* obstacles,
                      int nx, int ny)
{
  /* get column and row indices */
  int jj = get_global_id(0);
  int ii = get_global_id(1);
  const int N = nx*ny;

  /* determine indices of axis-direction neighbours
  ** respecting periodic boundary conditions (wrap around) */
  int y_n = (ii + 1) % ny;
  int x_e = (jj + 1) % nx;
  int y_s = (ii == 0) ? (ii + ny - 1) : (ii - 1);
  int x_w = (jj == 0) ? (jj + nx - 1) : (jj - 1);
  /* propagate densities to neighbouring cells, following
  ** appropriate directions of travel and writing into
  ** scratch space grid */
  tmp_cells[0*N + ii * nx + jj] = cells[0*N + ii  * nx + jj ]; /* central cell */
  tmp_cells[1*N + ii * nx + jj] = cells[1*N + ii  * nx + x_w]; /* east */
  tmp_cells[2*N + ii * nx + jj] = cells[2*N + y_s * nx + jj ]; /* north */
  tmp_cells[3*N + ii * nx + jj] = cells[3*N + ii  * nx + x_e]; /* west */
  tmp_cells[4*N + ii * nx + jj] = cells[4*N + y_n * nx + jj ]; /* south */
  tmp_cells[5*N + ii * nx + jj] = cells[5*N + y_s * nx + x_w]; /* north-east */
  tmp_cells[6*N + ii * nx + jj] = cells[6*N + y_s * nx + x_e]; /* north-west */
  tmp_cells[7*N + ii * nx + jj] = cells[7*N + y_n * nx + x_e]; /* south-west */
  tmp_cells[8*N + ii * nx + jj] = cells[8*N + y_n * nx + x_w]; /* south-east */
}

void reduce(
   __local  float*,
   __global float*);


kernel void total_velocity(
   global float* cells,
   global float* tmp_cells,
   global int* 	   obstacles,
   local  float*   local_sums,
   global float*   partial_sums,
   int nx, int ny, float omega)
{
  int ii = get_global_id(0);
  int local_id = get_local_id(0);
  const int N = nx*ny;

  const float c_sq_inv = 3.0; /* square of speed of sound */
  const float w0 = 4.0 / 9.0;  /* weighting factor */
  const float w1 = 1.0 / 9.0;  /* weighting factor */
  const float w2 = 1.0 / 36.0; /* weighting factor */


  float private_cell[NSPEEDS];
  for (int kk = 0; kk < NSPEEDS; kk++)
    private_cell[kk] = tmp_cells[kk*N + ii];

  float local_density = 0.0;
  for (int kk = 0; kk < NSPEEDS; kk++)
    local_density += private_cell[kk];

  /* x-component of velocity */
  float u_x = (private_cell[1]
             + private_cell[5]
             + private_cell[8]
             - (private_cell[3]
                + private_cell[6]
                + private_cell[7]))
             / local_density;

  /* compute y velocity component */
  float u_y = (private_cell[2]
             + private_cell[5]
             + private_cell[6]
             - (private_cell[4]
                + private_cell[7]
                + private_cell[8]))
             / local_density;

  /* velocity squared */
  float u_sq = u_x * u_x + u_y * u_y;

  /* directional velocity components */
  float d_equ[NSPEEDS];
  d_equ[1] =   u_x;        /* east       */
  d_equ[2] =         u_y;  /* north      */
  d_equ[3] = - u_x;        /* west       */
  d_equ[4] =       - u_y;  /* south      */
  d_equ[5] =   u_x + u_y;  /* north-east */
  d_equ[6] = - u_x + u_y;  /* north-west */
  d_equ[7] = - u_x - u_y;  /* south-west */
  d_equ[8] =   u_x - u_y;  /* south-east */

  /* zero velocity density: weight w0 */
  d_equ[0] = w0 * local_density * (1.0 - u_sq * 0.5 * c_sq_inv);
  d_equ[1] = w1 * local_density * (1.0 + d_equ[1] * c_sq_inv
                                   + (d_equ[1] * d_equ[1]) * (0.5 * c_sq_inv * c_sq_inv)
                                   - u_sq * (0.5 * c_sq_inv));
  d_equ[2] = w1 * local_density * (1.0 + d_equ[2] * c_sq_inv
                                   + (d_equ[2] * d_equ[2]) * (0.5 * c_sq_inv * c_sq_inv)
                                   - u_sq * (0.5 * c_sq_inv));
  d_equ[3] = w1 * local_density * (1.0 + d_equ[3] * c_sq_inv
                                   + (d_equ[3] * d_equ[3]) * (0.5 * c_sq_inv * c_sq_inv)
                                   - u_sq * (0.5 * c_sq_inv));
  d_equ[4] = w1 * local_density * (1.0 + d_equ[4] * c_sq_inv
                                   + (d_equ[4] * d_equ[4]) * (0.5 * c_sq_inv * c_sq_inv)
                                   - u_sq * (0.5 * c_sq_inv));
  // diagonal speeds: weight w2
  d_equ[5] = w2 * local_density * (1.0 + d_equ[5] * c_sq_inv
                                   + (d_equ[5] * d_equ[5]) * (0.5 * c_sq_inv * c_sq_inv)
                                   - u_sq * (0.5 * c_sq_inv));
  d_equ[6] = w2 * local_density * (1.0 + d_equ[6] * c_sq_inv
                                   + (d_equ[6] * d_equ[6]) * (0.5 * c_sq_inv * c_sq_inv)
                                   - u_sq * (0.5 * c_sq_inv));
  d_equ[7] = w2 * local_density * (1.0 + d_equ[7] * c_sq_inv
                                   + (d_equ[7] * d_equ[7]) * (0.5 * c_sq_inv * c_sq_inv)
                                   - u_sq * (0.5 * c_sq_inv));
  d_equ[8] = w2 * local_density * (1.0 + d_equ[8] * c_sq_inv
                                   + (d_equ[8] * d_equ[8]) * (0.5 * c_sq_inv * c_sq_inv)
                                   - u_sq * (0.5 * c_sq_inv));
  /* relaxation step */

  // use the same d_equ array to store new cell
  int obstacle = obstacles[ii];
  d_equ[0] = obstacle ?
       			cells[0*N + ii] :
        		private_cell[0]
        		+ omega
        		* (d_equ[0] - private_cell[0]);
  d_equ[1] = obstacle ?
        		private_cell[3] :
        		private_cell[1]
                       	+ omega
                       	* (d_equ[1] - private_cell[1]);
  d_equ[2] = obstacle ?
        		private_cell[4] :
        		private_cell[2]
                       	+ omega
                       	* (d_equ[2] - private_cell[2]);
  d_equ[3] = obstacle ?
        		private_cell[1] :
        		private_cell[3]
                       	+ omega
                       	* (d_equ[3] - private_cell[3]);
  d_equ[4] = obstacle ?
        		private_cell[2] :
        		private_cell[4]
                       	+ omega
                       	* (d_equ[4] - private_cell[4]);
  d_equ[5] = obstacle ?
        		private_cell[7] :
        		private_cell[5]
                       	+ omega
                       	* (d_equ[5] - private_cell[5]);
  d_equ[6] = obstacle ?
        		private_cell[8] :
        		private_cell[6]
                       	+ omega
                       	* (d_equ[6] - private_cell[6]);
  d_equ[7] = obstacle ?
        		private_cell[5] :
        		private_cell[7]
                       	+ omega
                       	* (d_equ[7] - private_cell[7]);
  d_equ[8] = obstacle ?
        		private_cell[6] :
        		private_cell[8]
                       	+ omega
                       	* (d_equ[8] - private_cell[8]);

  for (int kk = 0; kk < NSPEEDS; kk++)
    cells[kk*N + ii] = d_equ[ii];

  /* compute new velocity */
  local_density = 0.0;

  for (int kk = 0; kk < NSPEEDS; kk++)
    local_density += d_equ[kk];

  /* x-component of velocity */
  u_x = (d_equ[1]
             + d_equ[5]
             + d_equ[8]
             - (d_equ[3]
                + d_equ[6]
                + d_equ[7]))
             ;

  /* compute y velocity component */
  u_y = (d_equ[2]
             + d_equ[5]
             + d_equ[6]
             - (d_equ[4]
                + d_equ[7]
                + d_equ[8]))
             ;

  /* velocity squared */
  float vel = sqrt(u_x * u_x + u_y * u_y) / local_density;
  local_sums[local_id] = !obstacle * vel;
 
  barrier(CLK_LOCAL_MEM_FENCE);

  reduce(local_sums, partial_sums);
}


void reduce(
   __local  float*    local_sums,
   __global float*    partial_sums)
{
   int num_wrk_items  = get_local_size(0);
   int local_id       = get_local_id(0);
   int group_id       = get_group_id(0);

   float sum;
   int i;

   if (local_id == 0) {
      sum = 0.0f;

      for (i=0; i<num_wrk_items; i++) {
          sum += local_sums[i];
      }

      partial_sums[group_id] = sum;
   }
}
