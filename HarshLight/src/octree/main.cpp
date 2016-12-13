#include <stdlib.h>
#include <stdio.h>
#include <string>
#include <cstring>

void cudaBuildMalloc(int* colors, int* normals, int voxel_dim, int* res);
// int cudaBuild(cudaSurfaceObject_t colors, cudaSurfaceObject_t normals, uint32_t voxel_dim, int* tmp, int* result);
//int cudaBuild(int* colors, int* normals, uint32_t voxel_dim, int* tmp, int* result);
void printCudaInfo();

const uint32_t dim = 4;

static inline int combine(int a, int b, int c, int d) {
	a = a & 0xFF;
	b = b & 0xFF;
	c = c & 0xFF;
	d = d & 0xFF;
	
	return (a << 24) + (b << 16) + (c << 8) + d;
}
int main(int argc, char** argv) {
    printCudaInfo();

	int total = dim * dim * dim;
	int* c = (int*)malloc(sizeof(int) * (total + 5));
	int* n = (int*)malloc(sizeof(int) * (total + 5));
    memset(c, 0, sizeof(int) * (total + 5));
	
	for (int i = 0; i < total; i += 7) {
		int a = rand()%255 + 1;
		c[i] = combine(rand()%256, rand()%256, rand()%256, a);
		n[i] = combine(rand()%256, rand()%256, rand()%256, a);
	}
	
	int blocksize = 16;
	int* res = (int*)malloc(total * sizeof(int) * blocksize + 5);
	cudaBuildMalloc(c, n, dim, res);
	
	for (int i = 0; i < total; ++i) {
		printf("node: %d\nnormal: 0x%X\nfather: %d\nsons:%d\n", i, res[i * blocksize + 1], res[i * blocksize + 2], res[i * blocksize + 3]);
		
		for (int j = 0; j < 6; ++j) {
			printf("neightb %d: %d\n", j, res[i * blocksize + j + 4]);
		}
		printf("\n");
	}
    
    return 0;
}
