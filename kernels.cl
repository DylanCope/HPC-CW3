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
 
  float local_density = 0.0;

  for (int kk = 0; kk < NSPEEDS; kk++)
    local_density += tmp_cells[kk*N + ii];

  /* x-component of velocity */
  float u_x = (tmp_cells[1*N + ii]
             + tmp_cells[5*N + ii]
             + tmp_cells[8*N + ii]
             - (tmp_cells[3*N + ii]
                + tmp_cells[6*N + ii]
                + tmp_cells[7*N + ii])) 
             / local_density;

  /* compute y velocity component */
  float u_y = (tmp_cells[2*N + ii]
             + tmp_cells[5*N + ii]
             + tmp_cells[6*N + ii]
             - (tmp_cells[4*N + ii]
                + tmp_cells[7*N + ii]
                + tmp_cells[8*N + ii])) 
             / local_density;

  /* velocity squared */
  float u_sq = u_x * u_x + u_y * u_y;

  /* directional velocity components */
  float u[NSPEEDS];
  u[1] =   u_x;        /* east       */
  u[2] =         u_y;  /* north      */
  u[3] = - u_x;        /* west       */
  u[4] =       - u_y;  /* south      */
  u[5] =   u_x + u_y;  /* north-east */
  u[6] = - u_x + u_y;  /* north-west */
  u[7] = - u_x - u_y;  /* south-west */
  u[8] =   u_x - u_y;  /* south-east */

  /* equilibrium densities */
  float d_equ[NSPEEDS];
  /* zero velocity density: weight w0 */
  d_equ[0] = w0 * local_density * (1.0 - u_sq * 0.5 * c_sq_inv);
  d_equ[1] = w1 * local_density * (1.0 + u[1] * c_sq_inv
                                   + (u[1] * u[1]) * (0.5 * c_sq_inv * c_sq_inv)
                                   - u_sq * (0.5 * c_sq_inv));
  d_equ[2] = w1 * local_density * (1.0 + u[2] * c_sq_inv
                                   + (u[2] * u[2]) * (0.5 * c_sq_inv * c_sq_inv)
                                   - u_sq * (0.5 * c_sq_inv));
  d_equ[3] = w1 * local_density * (1.0 + u[3] * c_sq_inv
                                   + (u[3] * u[3]) * (0.5 * c_sq_inv * c_sq_inv)
                                   - u_sq * (0.5 * c_sq_inv));
  d_equ[4] = w1 * local_density * (1.0 + u[4] * c_sq_inv
                                   + (u[4] * u[4]) * (0.5 * c_sq_inv * c_sq_inv)
                                   - u_sq * (0.5 * c_sq_inv));
  // diagonal speeds: weight w2
  d_equ[5] = w2 * local_density * (1.0 + u[5] * c_sq_inv
                                   + (u[5] * u[5]) * (0.5 * c_sq_inv * c_sq_inv)
                                   - u_sq * (0.5 * c_sq_inv));
  d_equ[6] = w2 * local_density * (1.0 + u[6] * c_sq_inv
                                   + (u[6] * u[6]) * (0.5 * c_sq_inv * c_sq_inv)
                                   - u_sq * (0.5 * c_sq_inv));
  d_equ[7] = w2 * local_density * (1.0 + u[7] * c_sq_inv
                                   + (u[7] * u[7]) * (0.5 * c_sq_inv * c_sq_inv)
                                   - u_sq * (0.5 * c_sq_inv));
  d_equ[8] = w2 * local_density * (1.0 + u[8] * c_sq_inv
                                   + (u[8] * u[8]) * (0.5 * c_sq_inv * c_sq_inv)
                                   - u_sq * (0.5 * c_sq_inv));
  /* relaxation step */


  cells[0*N + ii] = obstacles[ii] ?
			cells[0*N + ii] :
			tmp_cells[0*N + ii]
			+ omega
			* (d_equ[0] - tmp_cells[0*N + ii]); 
  cells[1*N + ii] = obstacles[ii] ? 
			tmp_cells[3*N + ii] :
 			tmp_cells[1*N + ii]
                       	+ omega
                       	* (d_equ[1] - tmp_cells[1*N + ii]);
  cells[2*N + ii] = obstacles[ii] ? 
			tmp_cells[4*N + ii] :
 			tmp_cells[2*N + ii]
                       	+ omega
                       	* (d_equ[2] - tmp_cells[2*N + ii]);
  cells[3*N + ii] = obstacles[ii] ? 
			tmp_cells[1*N + ii] :
 			tmp_cells[3*N + ii]
                       	+ omega
                       	* (d_equ[3] - tmp_cells[3*N + ii]);
  cells[4*N + ii] = obstacles[ii] ? 
			tmp_cells[2*N + ii] :
 			tmp_cells[4*N + ii]
                       	+ omega
                       	* (d_equ[4] - tmp_cells[4*N + ii]);
  cells[5*N + ii] = obstacles[ii] ? 
			tmp_cells[7*N + ii] :
 			tmp_cells[5*N + ii]
                       	+ omega
                       	* (d_equ[5] - tmp_cells[5*N + ii]);
  cells[6*N + ii] = obstacles[ii] ? 
			tmp_cells[8*N + ii] :
 			tmp_cells[6*N + ii]
                       	+ omega
                       	* (d_equ[6] - tmp_cells[6*N + ii]);
  cells[7*N + ii] = obstacles[ii] ? 
			tmp_cells[5*N + ii] :
 			tmp_cells[7*N + ii]
                       	+ omega
                       	* (d_equ[7] - tmp_cells[7*N + ii]);
  cells[8*N + ii] = obstacles[ii] ? 
			tmp_cells[6*N + ii] :
 			tmp_cells[8*N + ii]
                       	+ omega
                       	* (d_equ[8] - tmp_cells[8*N + ii]);
 
  /* compute new velocity */
  local_density = 0.0;

  for (int kk = 0; kk < NSPEEDS; kk++)
    local_density += cells[kk*N + ii];

  /* x-component of velocity */
  u_x = (cells[1*N + ii]
             + cells[5*N + ii]
             + cells[8*N + ii]
             - (cells[3*N + ii]
                + cells[6*N + ii]
                + cells[7*N + ii])) 
             ;

  /* compute y velocity component */
  u_y = (cells[2*N + ii]
             + cells[5*N + ii]
             + cells[6*N + ii]
             - (cells[4*N + ii]
                + cells[7*N + ii]
                + cells[8*N + ii])) 
             ;

  /* velocity squared */
  float vel = sqrt(u_x * u_x + u_y * u_y) / local_density;

  float mask = obstacles[ii] ? 0.0 : 1.0;
  local_sums[local_id] = mask * vel;
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
