
#include "../common/cpu_bitmap.h"

#define DIM 2080
#define sigma 10
#define rho 28
#define beta 8.0/3.0
#define scale 28.0

struct color {
    int red;
    int yellow;
    int blue;
    int _;
};

struct point_idx {
    int x, y;
};

int cal_ind(float x) {
    return int((DIM/2.0)*(x/scale) + (DIM/2.0));
}

void line( unsigned char *ptr, color c, point_idx p1, point_idx p2) {
    bool is_x_longer = abs(p2.x - p1.x) > abs(p2.y - p1.y);

    int start = is_x_longer ? min(p1.x, p2.x) : min(p1.y, p2.y);
    int end = is_x_longer ? max(p1.x, p2.x) : max(p1.y, p2.y);
    if (start == end) return;

    for (int n=start; n<end; n++) {
        int x, y;
        if (is_x_longer) {
            x = n;
            y = (p2.y - p1.y) / (1.0*(p2.x - p1.x)) * (x - p1.x) + p1.y;
        }
        else {
            y = n;
            x = (p2.x - p1.x) / (1.0*(p2.y - p1.y)) * (y - p1.y) + p1.x;
        }
        int offset = x + y * DIM;
        ptr[offset*4 + 0] = c.red;
        ptr[offset*4 + 1] = c.yellow;
        ptr[offset*4 + 2] = c.blue;
        ptr[offset*4 + 3] = c._;
    }
}


void kernel( unsigned char *ptr ) {
    float t_max = 80;
    float dt = 0.00007;
    float t = 0;

    float x = 1.0;
    float y = 1.0;
    float z = 1.0;

    int xid1 = cal_ind(x);
    int yid1 = cal_ind(y);
    int xid2, yid2;

    for (int i=0; i<int(t_max/dt); i++) {
        t = t + dt;
        float fx1 = sigma * (y - x);
        float fy1 = x * (rho - z) - y;
        float fz1 = x * y - beta * z;

        float x1 = x + 0.5*fx1*dt;
        float y1 = y + 0.5*fy1*dt;
        float z1 = z + 0.5*fz1*dt;

        float fx2 = sigma * (y1 - x1);
        float fy2 = x1 * (rho - z1) - y1;
        float fz2 = x1 * y1 - beta * z1;

        float x2 = x + 0.5*fx2*dt;
        float y2 = y + 0.5*fy2*dt;
        float z2 = z + 0.5*fz2*dt;

        float fx3 = sigma * (y2 - x2);
        float fy3 = x2 * (rho - z2) - y2;
        float fz3 = x2 * y2 - beta * z2;

        float x3 = x + fx3*dt;
        float y3 = y + fy3*dt;
        float z3 = z + fz3*dt;

        float fx4 = sigma * (y3 - x3);
        float fy4 = x3 * (rho - z3) - y3;
        float fz4 = x3 * y3 - beta * z3;

        x = x3 + (fx1 + 2*fx2 + 2*fx3 + fx4)*dt/6.0;
        y = y3 + (fy1 + 2*fy2 + 2*fy3 + fy4)*dt/6.0;
        z = z3 + (fz1 + 2*fz2 + 2*fz3 + fz4)*dt/6.0;


    //    x = x + (sigma * (y - (x_ + x)/2.0))*dt; 
    //     y = y + (x * (rho - z) - (y_ + y)/2.0)*dt;
    //     z = z + (x * y - beta * (z_ + z)/2.0)*dt;
        // color c = {128, int(255*(sin(5*t/t_max)+1)/2.0), 128, 255};
        // color c = {int(255 * t / t_max), int(255*(sin(5*t/t_max)+1)/2.0), int(255 - 255 * t / t_max), 255};

        // 04
        color c = {int(255 * t / t_max), 0, int(255 - 255 * t / t_max), 255};

        xid2 = cal_ind(x);
        yid2 = cal_ind(y);
        point_idx p1 = {xid1, yid1};
        point_idx p2 = {xid2, yid2};
        line(ptr, c, p1, p2);

        xid1 = xid2;
        yid1 = yid2;

        // int xid = cal_ind(x);
        // int yid = cal_ind(y);
        // int offset = xid + yid * DIM;

        // ptr[offset*4 + 0] = c.red;
        // ptr[offset*4 + 1] = c.yellow;
        // ptr[offset*4 + 2] = c.blue;
        // ptr[offset*4 + 3] = c._;
    }
}

int main( void ) {
    CPUBitmap bitmap( DIM, DIM );
    unsigned char *ptr = bitmap.get_ptr();

    kernel( ptr );
                              
    bitmap.display_and_exit();
}