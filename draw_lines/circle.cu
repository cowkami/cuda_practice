/* Copyright 1993-2010 NVIDIA Corporation.  All rights reserved.
 *
 * NVIDIA Corporation and its licensors retain all intellectual property and 
 * proprietary rights in and to this software and related documentation. 
 * Any use, reproduction, disclosure, or distribution of this software 
 * and related documentation without an express license agreement from
 * NVIDIA Corporation is strictly prohibited.
 *
 * Please refer to the applicable NVIDIA end user license agreement (EULA) 
 * associated with this source code for terms and conditions that govern 
 * your use of this NVIDIA software.
 * 
 */
 

#include "../common/book.h"
#include "../common/cpu_bitmap.h"

#define DIM 2000

__device__ int straight_line( float x, float y, float grad, float bias, float line_width ) {
    if (pow(grad*x + bias - y, 2) <= pow(line_width, 2)) return 1;
    return 0;
}

__device__ int circle_line( float x, float y , float c_x, float c_y, float radius, float line_width ) {
    float eq = pow(x - c_x, 2) + pow(y - c_y, 2);
    if (pow(radius - line_width, 2) <= eq && eq <= pow(radius + line_width, 2)) return 1;
    return 0;
}
__global__ void kernel( unsigned char *ptr ) {
    // map from blockIdx to pixel position
    int x = blockIdx.x;
    int y = blockIdx.y;
    int offset = x + y * gridDim.x;

    const float scale = 2000;
    float jx = scale * (float)(DIM/2 - x)/(DIM/2);
    float jy = scale * (float)(DIM/2 - y)/(DIM/2);

    // now calculate the value at that position
    // int pixel_value = straight_line( x, y, 1, 0, 2 );
    // int pixel_value = circle_line( x, y, 1000, 1000, 100, 2);
    // int pixel_value = straight_line( x, y, 1, 0, 2 ) || circle_line( x, y, 1000, 1000, 100, 2);
    ptr[offset*4 + 0] = 255 * pixel_value;
    ptr[offset*4 + 1] = 0;
    ptr[offset*4 + 2] = 0;
    ptr[offset*4 + 3] = 0;
}

// globals needed by the update routine
struct DataBlock {
    unsigned char   *dev_bitmap;
};

int main( void ) {
    DataBlock   data;
    CPUBitmap bitmap( DIM, DIM, &data );
    unsigned char    *dev_bitmap;

    HANDLE_ERROR( cudaMalloc( (void**)&dev_bitmap, bitmap.image_size() ) );
    data.dev_bitmap = dev_bitmap;

    dim3    grid(DIM,DIM);
    kernel<<<grid,1>>>( dev_bitmap );

    HANDLE_ERROR( cudaMemcpy( bitmap.get_ptr(), dev_bitmap,
                              bitmap.image_size(),
                              cudaMemcpyDeviceToHost ) );
                              
    HANDLE_ERROR( cudaFree( dev_bitmap ) );
                              
    bitmap.display_and_exit();
}

