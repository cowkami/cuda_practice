#include "../common/book.h"
#include "../common/cpu_bitmap.h"

#define DIM 1080
#define sigma 10
#define rho 28
#define beta 8.0/3.0
#define scale 30.0

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
    int t_max = 80;
    float dt = 0.001;

    float x = 1.0;
    float y = 1.0;
    float z = 1.0;

    int xid1 = cal_ind(x);
    int yid1 = cal_ind(y);
    int xid2, yid2;

    for (int i=0; i<int(t_max/dt); i++) {
        x = x + (sigma * (y - x))*dt;
        y = y + (x * (rho - z) - y)*dt;
        z = z + (x * y - beta * z)*dt;
        color c = {int(255 * z / 50), 128, int(255 - 255 * z / 50), 255};
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

        // ptr[offset*4 + 0] = 255;
        // ptr[offset*4 + 1] = int(155 * z / 50);
        // ptr[offset*4 + 2] = int(255 * z / 50);
        // ptr[offset*4 + 3] = 255;
    }
}

int main( void ) {
    CPUBitmap bitmap( DIM, DIM );
    unsigned char *ptr = bitmap.get_ptr();

    kernel( ptr );
                              
    bitmap.display_and_exit();
}