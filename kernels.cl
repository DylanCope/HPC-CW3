#pragma OPENCL EXTENSION cl_khr_fp64 : enable

#define NSPEEDS         9

typedef struct
{
  float speeds[NSPEEDS];
} t_speed;

kernel void accelerate_flow(global t_speed* cells,
                            global int* obstacles,
                            int nx, int ny,
                            float density, float accel)
{
  /* compute weighting factors */
  float w1 = density * accel / 9.0;
  float w2 = density * accel / 36.0;

  /* modify the 2nd row of the grid */
  int ii = ny - 2;

  /* get column index */
  int jj = get_global_id(0);

  /* if the cell is not occupied and
  ** we don't send a negative density */
  if (!obstacles[ii * nx + jj]
      && (cells[ii * nx + jj].speeds[3] - w1) > 0.0
      && (cells[ii * nx + jj].speeds[6] - w2) > 0.0
      && (cells[ii * nx + jj].speeds[7] - w2) > 0.0)
  {
    /* increase 'east-side' densities */
    cells[ii * nx + jj].speeds[1] += w1;
    cells[ii * nx + jj].speeds[5] += w2;
    cells[ii * nx + jj].speeds[8] += w2;
    /* decrease 'west-side' densities */
    cells[ii * nx + jj].speeds[3] -= w1;
    cells[ii * nx + jj].speeds[6] -= w2;
    cells[ii * nx + jj].speeds[7] -= w2;
  }
}

kernel void propagate(global t_speed* cells,
                      global t_speed* tmp_cells,
                      global int* obstacles,
                      int nx, int ny)
{
  /* get column and row indices */
  int jj = get_global_id(0);
  int ii = get_global_id(1);

  /* determine indices of axis-direction neighbours
  ** respecting periodic boundary conditions (wrap around) */
  int y_n = (ii + 1) % ny;
  int x_e = (jj + 1) % nx;
  int y_s = (ii == 0) ? (ii + ny - 1) : (ii - 1);
  int x_w = (jj == 0) ? (jj + nx - 1) : (jj - 1);
  /* propagate densities to neighbouring cells, following
  ** appropriate directions of travel and writing into
  ** scratch space grid */
  tmp_cells[ii  * nx + jj ].speeds[0] = cells[ii * nx + jj].speeds[0]; /* central cell, no movement */
  tmp_cells[ii  * nx + x_e].speeds[1] = cells[ii * nx + jj].speeds[1]; /* east */
  tmp_cells[y_n * nx + jj ].speeds[2] = cells[ii * nx + jj].speeds[2]; /* north */
  tmp_cells[ii  * nx + x_w].speeds[3] = cells[ii * nx + jj].speeds[3]; /* west */
  tmp_cells[y_s * nx + jj ].speeds[4] = cells[ii * nx + jj].speeds[4]; /* south */
  tmp_cells[y_n * nx + x_e].speeds[5] = cells[ii * nx + jj].speeds[5]; /* north-east */
  tmp_cells[y_n * nx + x_w].speeds[6] = cells[ii * nx + jj].speeds[6]; /* north-west */
  tmp_cells[y_s * nx + x_w].speeds[7] = cells[ii * nx + jj].speeds[7]; /* south-west */
  tmp_cells[y_s * nx + x_e].speeds[8] = cells[ii * nx + jj].speeds[8]; /* south-east */
}

kernel void rebound(global t_speed* cells,
		    global t_speed* tmp_cells,
		    global int* obstacles,
		    int nx, int ny,
 		    float omega)
{
  const float c_sq_inv =  3.0; /* square of speed of sound */
  const float w0 = 4.0 / 9.0;  /* weighting factor */
  const float w1 = 1.0 / 9.0;  /* weighting factor */
  const float w2 = 1.0 / 36.0; /* weighting factor */

  int jj = get_global_id(0);
  int ii = get_global_id(1);
  if (obstacles[ii * nx + jj])
  {
    cells[ii * nx + jj].speeds[1] = tmp_cells[ii * nx + jj].speeds[3];
    cells[ii * nx + jj].speeds[2] = tmp_cells[ii * nx + jj].speeds[4];
    cells[ii * nx + jj].speeds[3] = tmp_cells[ii * nx + jj].speeds[1];
    cells[ii * nx + jj].speeds[4] = tmp_cells[ii * nx + jj].speeds[2];
    cells[ii * nx + jj].speeds[5] = tmp_cells[ii * nx + jj].speeds[7];
    cells[ii * nx + jj].speeds[6] = tmp_cells[ii * nx + jj].speeds[8];
    cells[ii * nx + jj].speeds[7] = tmp_cells[ii * nx + jj].speeds[5];
    cells[ii * nx + jj].speeds[8] = tmp_cells[ii * nx + jj].speeds[6];
  }
  else
  {
    /* compute local density total */
    float local_density = 0.0;

    for (int kk = 0; kk < NSPEEDS; kk++)
    {
      local_density += tmp_cells[ii * nx + jj].speeds[kk];
    }

    /* compute x velocity component */
    float u_x = (tmp_cells[ii * nx + jj].speeds[1]
                  + tmp_cells[ii * nx + jj].speeds[5]
                  + tmp_cells[ii * nx + jj].speeds[8]
                  - (tmp_cells[ii * nx + jj].speeds[3]
                     + tmp_cells[ii * nx + jj].speeds[6]
                     + tmp_cells[ii * nx + jj].speeds[7]))
                 / local_density;
    /* compute y velocity component */
    float u_y = (tmp_cells[ii * nx + jj].speeds[2]
                  + tmp_cells[ii * nx + jj].speeds[5]
                  + tmp_cells[ii * nx + jj].speeds[6]
                  - (tmp_cells[ii * nx + jj].speeds[4]
                     + tmp_cells[ii * nx + jj].speeds[7]
                     + tmp_cells[ii * nx + jj].speeds[8]))
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
    for (int kk = 0; kk < NSPEEDS; kk++)
    {
      cells[ii * nx + jj].speeds[kk] = tmp_cells[ii * nx + jj].speeds[kk]
                                              + omega
                                              * (d_equ[kk] - tmp_cells[ii * nx + jj].speeds[kk]);
    }
  } 
}

void reduce(                                          
   __local  float*,                          
   __global float*); 


kernel void total_velocity(
   global t_speed* cells, 
   global int* 	   obstacles,                                                       
   local  float*   local_sums,                          
   global float*   partial_sums,
   int             n)                        
{                                                          
   int local_size = get_local_size(0);
   int ii  = get_global_id(0) + get_local_id(0);                   

   float accum = 0.0f;                              

   if (!obstacles[ii])
     for (int kk = 0; kk < NSPEEDS; kk++)  
       accum += cells[ii].speeds[kk];  

   int local_id = get_local_id(0);
   local_sums[local_id] = accum;
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
