
typedef struct{
    float x;
    float y;
    float z;
} floatVox;

typedef struct{
    int x;
    int y;
    int z;
} intVox;


floatVox scale(floatVox, float);
floatVox add(floatVox, floatVox);
floatVox ray_normalize(floatVox);
floatVox ray_cross(floatVox, floatVox);

int inside(intVox);
int index(int, int, int);
int inside_float(floatVox);

float value_at(floatVox, const __global unsigned char*);

// floatVox utilities
floatVox ray_cross(floatVox a, floatVox b){
    floatVox c;
    c.x = a.y*b.z - a.z*b.y;
    c.y = a.z*b.x - a.x*b.z;
    c.z = a.x*b.y - a.y*b.x;

    return c;
}

floatVox ray_normalize(floatVox v){
    float l = sqrt(v.x*v.x + v.y*v.y + v.z*v.z);
    v.x /= l;
    v.y /= l;
    v.z /= l;

    return v;
}

floatVox add(floatVox a, floatVox b){
    a.x += b.x;
    a.y += b.y;
    a.z += b.z;

    return a;
}

floatVox scale(floatVox a, float b){
    a.x *= b;
    a.y *= b;
    a.z *= b;

    return a;
}
// Checks if position is inside the volume (floatVox and intVox versions)
int inside_float(floatVox pos){
    int x = (pos.x >= 0 && pos.x < (512)-1);
    int y = (pos.y >= 0 && pos.y < (512)-1);
    int z = (pos.z >= 0 && pos.z < (512)-1);

    return x && y && z;
}

int inside(intVox pos){
    int x = (pos.x >= 0 && pos.x < 512);
    int y = (pos.y >= 0 && pos.y < 512);
    int z = (pos.z >= 0 && pos.z < 512);
    return x && y && z;
}

// Indexing function (note the argument order)
int index(int z, int y, int x){
    return (z*512*512)
            + y*(512) + x;
}

// Trilinear interpolation
float value_at(floatVox pos, const __global unsigned char* data){
    if(!inside_float(pos)){
        return 0;
    }

    int x = floor(pos.x);
    int y = floor(pos.y);
    int z = floor(pos.z);

    int x_u = ceil(pos.x);
    int y_u = ceil(pos.y);
    int z_u = ceil(pos.z);

    float rx = pos.x - x;
    float ry = pos.y - y;
    float rz = pos.z - z;

    float a0 = rx*data[index(z,y,x)] + (1-rx)*data[index(z,y,x_u)];
    float a1 = rx*data[index(z,y_u,x)] + (1-rx)*data[index(z,y_u,x_u)];
    float a2 = rx*data[index(z_u,y,x)] + (1-rx)*data[index(z_u,y,x_u)];
    float a3 = rx*data[index(z_u,y_u,x)] + (1-rx)*data[index(z_u,y_u,x_u)];

    float b0 = ry*a0 + (1-ry)*a1;
    float b1 = ry*a2 + (1-ry)*a3;

    float c0 = rz*b0 + (1-rz)*b1;


    return c0;
}

__kernel
void raycast(const __global unsigned char* data, const __global unsigned char* region, __global unsigned char* image){
    const int idx = get_global_id(0);
    const int idy = get_global_id(1);

    const int y = idy - (512/2);
    const int x = idx - (512/2);
    floatVox z_axis = {.x=0, .y=0, .z = 1};
    floatVox forward = {.x=-1, .y=-1, .z=-1};
    floatVox camera = {.x=1000, .y=1000, .z=1000};

    floatVox right = ray_cross(forward, z_axis);
    floatVox up = ray_cross(right, forward);

    up = ray_normalize(up);
    right = ray_normalize(right);
    forward = ray_normalize(forward);

    float fov = 3.14/4;
    float step_size = 0.5;
    float pixel_width = tan(fov/2.0)/(512/2);

    //Do the raycasting
    floatVox screen_center = add(camera, forward);
    floatVox ray = add(add(screen_center, scale(right, x*pixel_width)), scale(up, y*pixel_width));
    ray = add(ray, scale(camera, -1));
    ray = ray_normalize(ray);
    floatVox pos = camera;

    float color = 0;
    for (int i = 0; 255 > color && (5000 > i); ++i){
        pos = add(pos, scale(ray, step_size));
        if(!inside_float(pos)){
            continue;
        }
        int r = value_at(pos, region);
        color += value_at(pos, data)*(0.01+r);
    }
    image[idx + idy*512] = min(color, 255.f);
    return;
}
