#define DIM1 4
#define DIM2 500
#define DIM3 1000
#define DIM4 1000
#define DIM2 4
#define DIM3 4
#define DIM4 4
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#define _OPENMP 1
#include <zfp.h>
#include <time.h>

void fill_array(float ****array, float sc1, float sc2, float sc3, float sc4) {
    float *b1=(float*)malloc(sizeof(float)*DIM4);
        float *b2=(float*)malloc(sizeof(float)*DIM3);
            for (int k = 0; k < DIM3; k++) 
  b2[k] = cos(2 * M_PI * sc3 * k / DIM3);

            for (int k = 0; k < DIM4; k++) 
  b1[k] = cos(2 * M_PI * sc4 * k / DIM4);
    for (int i = 0; i < DIM1; i++) {
        float ar1 = cos(2 * M_PI * sc1 * i / DIM1);
        for (int j = 0; j < DIM2; j++) {
            float ar2 = cos(2 * M_PI * sc2 * j / DIM2);
            for (int k = 0; k < DIM3; k++) {
                for (int l = 0; l < DIM4; l++) {
                    array[i][j][k][l] = ar1 * ar2 * b1[l]*b2[k];
                }
            }
        }
    }
    free(b1);
    free(b2);
}


float ****allocate_4d_array(int dim1, int dim2, int dim3, int dim4) {
    // Allocate a single large block for all data
    float *block = (float *)malloc(dim1 * dim2 * dim3 * dim4 * sizeof(float));
    if (block == NULL) {
        return NULL;
    }

    // Allocate pointers for the first dimension
    float ****array = (float ****)malloc(dim1 * sizeof(float ***));
    if (array == NULL) {
        free(block);
        return NULL;
    }

    for (int i = 0; i < dim1; i++) {
        array[i] = (float ***)malloc(dim2 * sizeof(float **));
        if (array[i] == NULL) {
            // Free previously allocated memory
            while (i-- > 0) {
                free(array[i]);
            }
            free(array);
            free(block);
            return NULL;
        }

        for (int j = 0; j < dim2; j++) {
            array[i][j] = (float **)malloc(dim3 * sizeof(float *));
            if (array[i][j] == NULL) {
                // Free previously allocated memory
                while (j-- > 0) {
                    free(array[i][j]);
                }
                while (i-- > 0) {
                    free(array[i]);
                }
                free(array);
                free(block);
                return NULL;
            }

            for (int k = 0; k < dim3; k++) {
                array[i][j][k] = &block[i * dim2 * dim3 * dim4 + j * dim3 * dim4 + k * dim4];
            }
        }
    }

    return array;
}

void free_4d_array(float ****array, int dim1, int dim2, int dim3) {
    if (array == NULL) {
        return;
    }

    // Free the block of data
    free(array[0][0][0]);

    // Free the pointer arrays
    for (int i = 0; i < dim1; i++) {
        for (int j = 0; j < dim2; j++) {
            free(array[i][j]);
        }
        free(array[i]);
    }
    free(array);
}
void print_time(char *str,const struct timespec t1, const struct timespec t2){
 double diff=(t2.tv_nsec - t1.tv_nsec)/1000000000.0;
    diff+=t2.tv_sec - t1.tv_sec;
    fprintf(stderr,"%s %g\n",str,diff);
}

int main() {
    // Allocate the 4-D 
    float ****array = allocate_4d_array(DIM1,DIM2,DIM3,DIM4);
    float ****compare = allocate_4d_array(DIM1,DIM2,DIM3,DIM4);
    if (array == NULL) {
        perror("Memory allocation failed");
        return 1;
    }
    struct timespec time1,time2,time3,time4,time5,time6,time7,time8;



    // Call the fill_array function
    //
    clock_gettime(CLOCK_MONOTONIC, &time1);
    fill_array(array, 0.2f, 3.0f, 2.0f, 3.0f);
    clock_gettime(CLOCK_MONOTONIC, &time2);
    zfp_stream *zfp = zfp_stream_open(NULL);
    zfp_field *field=zfp_field_alloc(),*field2=zfp_field_alloc();
    zfp_field_set_type(field, zfp_type_float);
    zfp_field_set_size_4d(field, DIM1,DIM2,DIM3,DIM4);
    zfp_field_set_pointer(field, &(array[0][0][0][0]));
    zfp_field_set_type(field2, zfp_type_float);
    zfp_field_set_size_4d(field2, DIM1,DIM2,DIM3,DIM4);
    zfp_field_set_pointer(field2, &(compare[0][0][0][0]));

    zfp_stream_set_precision(zfp, 10);
    size_t bufsize=zfp_stream_maximum_size(zfp,field);
    fprintf(stderr,"BUFSIZE=%lld\n",bufsize);
    void *buffer =(void*) malloc(bufsize);
    bitstream *stream = stream_open(buffer, bufsize);
    zfp_stream_set_bit_stream(zfp, stream);
    int *n,*blocks;
    n=(int*)malloc(sizeof(int)*4);
    n[0]=DIM4; n[1]=DIM3;n[2]=DIM2;n[3]=DIM1;
    double blocks_per=1000.*1000./3./64.;
zfp_blocks *zfp_b=zfp_optimal_parts_from_size(4,n,blocks_per,1);
    fprintf(stderr,"Avout to coompress\n");
    fprintf(stderr,"ORIGINAL  %f \n",array[0][0][0][0]);

    clock_gettime(CLOCK_MONOTONIC, &time3);
    //fprintf(stderr,"COMPRESSS %lld\n",zfp_compress(zfp,field));
        fflush(stderr);
    stream_flush(stream);
    stream_rewind(stream);
    clock_gettime(CLOCK_MONOTONIC, &time4);
    //fprintf(stderr,"DECOMPRESSS %lld\n",zfp_decompress(zfp,field2));
    clock_gettime(CLOCK_MONOTONIC, &time5);
    fflush(stderr);
    fprintf(stderr,"xxx  %f \n",array[0][0][0][0]);
 //   fprintf(stderr,"AAAP %lld\n",zfp_omp_compress(zfp,field,16,zfp_b));

   fprintf(stderr,"DOING SINGLE\n");
    fprintf(stderr,"NNN OMP %lld\n",zfp_blocks_compress_single_stream(zfp,field,16,blocks_per,1));
    
    fflush(stderr);
    stream_flush(stream);
    stream_rewind(stream);
    clock_gettime(CLOCK_MONOTONIC, &time6);
    fprintf(stderr,"before decompress\n");
    fflush(stderr);
     // fprintf(stderr,"DECOMPRESSS OMP %lld\n",zfp_omp_decompress(zfp,field2,16,zfp_b));
    stream_rewind(stream);

    fprintf(stderr,"DECOMPRESSS SINGLE %lld\n",zfp_blocks_decompress_single_stream(zfp,field2,16));
        fprintf(stderr,"yyy  %f \n",compare[0][0][0][0]);

    fprintf(stderr,"DECOMPRESSED OMP %f \n",compare[0][0][0][0]);


    for(int i=0; i < 4; i++){
    for(int j=0; j < 4; j++){
    for(int k=0; k < 4; k++){
    for(int l=0; l < 4; l++){
	     fprintf(stderr, "COMPARE %d %d %d %d %f \n",i,j,k,l,compare[i][j][k][l]);
    }}}}


    clock_gettime(CLOCK_MONOTONIC, &time7);
    // Free the allocated memory
    free_4d_array(array,DIM1,DIM2,DIM3);
    free_4d_array(compare,DIM1,DIM2,DIM3);

    zfp_blocks_free(zfp_b);
    zfp_field_free(field);
    stream_close(stream);
    zfp_stream_close(zfp);
    free(buffer);
    free(n);
    clock_gettime(CLOCK_MONOTONIC, &time8);

    print_time("ALLOCA",time1,time2);
    print_time("ZFP-IN",time2,time3);
    print_time("COMP-S",time3,time4);
    print_time("DECO-S",time4,time5);
    print_time("COMP-P",time5,time6);
    print_time("DECO-P",time6,time7);
    print_time("FREE-M",time7,time8);


   

    return 0;
}
