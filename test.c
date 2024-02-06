#define DIM1 8
#define DIM2 100
#define DIM3 100
#define DIM4 146
//#define DIM2 12
//#define DIM3 12
//#define DIM4 12
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#define _OPENMP 1
#include <zfp.h>
#include <time.h>
#include <string.h>

typedef struct {
  float decompress;
  float compress;
  char compare_string[1024];
} compare_results;


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
}

double microseconds_calc(struct timespec time2, struct timespec time1){
    // Calculate difference
    long diff_sec = time1.tv_sec - time2.tv_sec;
    long diff_nsec = time1.tv_nsec - time2.tv_nsec;

    // Borrow one second from seconds if necessary
    if (diff_nsec < 0) {
        diff_nsec += 1000000000; // 1 second = 1,000,000,000 nanoseconds
        diff_sec--;
    }

    // Convert to microseconds
    long diff_microseconds = diff_sec * 1000000 + diff_nsec / 1000;
    return (double)diff_microseconds/1000./1000.;
}

compare_results compare_function(float ****input,float ****output){
  compare_results res;
  int p1=4,p2=54,p3=45,p4=31;
  float max_err=0.;
  float avg_err=0.;
  for(int i4=0; i4 < DIM4; i4++)
   for(int i3=0; i3 < DIM3; i3++)
     for(int i2=0; i2 < DIM2; i2++)
       for(int i1=0;i1 < DIM1; i1++){
          float diff=fabs(input[p1][p2][p3][p4]-output[p1][p2][p3][p4]);
          avg_err+=diff;
          if(diff>max_err) max_err=diff;

       }
  sprintf(res.compare_string,"%d %d %d %d MAX_ERR=%f AVG=ERR=%f \n",p1,p2,p3,p4,
    max_err,avg_err/(DIM1*DIM2*DIM3*DIM4));
  return res;
}
zfp_stream *create_stream(zfp_field *field){

    zfp_stream *zfp = zfp_stream_open(NULL);
    zfp_stream_set_accuracy(zfp, .02);
    return zfp;
}

zfp_field* zfp_field_from_floats(float ****array){
    zfp_field *field=zfp_field_alloc();
    zfp_field_set_type(field, zfp_type_float);
    zfp_field_set_size_4d(field, DIM4,DIM3,DIM2,DIM1);
    zfp_field_set_pointer(field, &(array[0][0][0][0]));
    return field;
}

compare_results test_single_thread_compression(float ****input, float ****output){
    zfp_field *inz =zfp_field_from_floats(input), 
              *outz=zfp_field_from_floats(output);
     struct timespec time1,time2,time3;
    zfp_stream *zfp=create_stream(inz);
    size_t nsz=zfp_stream_maximum_size(zfp,inz);
    void *buffer=(void*)malloc(zfp_stream_maximum_size(zfp,inz));  
    bitstream *bstream=stream_open(buffer,nsz);
    zfp_stream_set_bit_stream(zfp,bstream);

    clock_gettime(CLOCK_MONOTONIC, &time1);
    fprintf(stderr,"return compress %lld \n",zfp_compress(zfp,inz));
        stream_rewind(zfp->stream);
    clock_gettime(CLOCK_MONOTONIC, &time2);
    fprintf(stderr,"return decompress %lld \n",zfp_decompress(zfp,outz));
    clock_gettime(CLOCK_MONOTONIC, &time3);

    compare_results res=compare_function(input,output);
    res.compress=microseconds_calc(time1,time2);
    res.decompress=microseconds_calc(time2,time3);
    stream_close(bstream);

    zfp_field_free(inz);

    zfp_field_free(outz);


    zfp_stream_close(zfp);
        free(buffer);
        return res;
}

compare_results test_block_compression_single_stream(float ****input, float ****output){
    zfp_field *inz =zfp_field_from_floats(input), 
              *outz=zfp_field_from_floats(output);
    zfp_stream *zfp_in = create_stream(inz);
    zfp_stream *zfp_out= create_stream(outz);

    struct timespec time1, time2, time3;
    clock_gettime(CLOCK_MONOTONIC, &time1);
    fprintf(stderr,"before compress \n");
    size_t buf_size=zfp_blocks_compress_single_stream(zfp_in,inz,16, 1000.*1000./3./64.,1);
    clock_gettime(CLOCK_MONOTONIC, &time2);

    void *buf_out=malloc(buf_size);
    stream_rewind(zfp_in->stream);
    memcpy(buf_out,(const void*)stream_data(zfp_in->stream),buf_size);

    bitstream *stream=stream_open(buf_out,buf_size);
    zfp_stream_set_bit_stream(zfp_out,stream);
    free(stream_data(zfp_in->stream));
    stream_close(zfp_in->stream);
    zfp_stream_close(zfp_in);
    fprintf(stderr,"before decompress \n");

    clock_gettime(CLOCK_MONOTONIC, &time2);
    size_t decompress_loc=zfp_blocks_decompress_single_stream(zfp_out,outz, 16);
    clock_gettime(CLOCK_MONOTONIC, &time3);

    compare_results res=compare_function(input,output);
    res.compress=microseconds_calc(time1,time2);
    res.decompress=microseconds_calc(time2,time3);
    compare_function(input,output);

    zfp_field_free(inz);
    zfp_field_free(outz);
    free(buf_out);
    stream_close(zfp_out->stream);
    zfp_stream_close(zfp_out);
    return res;
}

compare_results test_block_compression_multi_stream(float ****input, float ****output){
    zfp_field *inz =zfp_field_from_floats(input), 
              *outz=zfp_field_from_floats(output);
    zfp_stream *zfp=create_stream(inz);
    
     struct timespec time1,time2,time3;
    clock_gettime(CLOCK_MONOTONIC, &time1);

    zfp_streams *zstreams=zfp_blocks_compress_multi(zfp,inz,16, 1000.*1000./3./64.,1);
    clock_gettime(CLOCK_MONOTONIC, &time2);
        stream_rewind(zfp->stream);

    zfp_blocks_decompress_multi_stream(zfp,outz,zstreams, 16);
    clock_gettime(CLOCK_MONOTONIC, &time3);

    compare_results res=compare_function(input,output);
    res.compress=microseconds_calc(time1,time2);
    res.decompress=microseconds_calc(time2,time3);
    zfp_streams_free(zstreams);
    zfp_field_free(inz);
    zfp_field_free(outz);
    free(stream_data(zfp->stream));
    stream_close(zfp->stream);

    zfp_stream_close(zfp);
    return res;
}

int main() {
    // Allocate the 4-D 
    float ****array = allocate_4d_array(DIM1,DIM2,DIM3,DIM4);
    float ****compare = allocate_4d_array(DIM1,DIM2,DIM3,DIM4);
    if (array == NULL || compare ==NULL) {
        perror("Memory allocation failed");
        return 1;
    }


    fill_array(array, 0.2f, 3.0f, 2.0f, 3.0f);
    zfp_stream *zfp = zfp_stream_open(NULL);

    //compare_results single_thread=test_single_thread_compression(array,compare);
    compare_results single_stream_multi_parts=test_block_compression_single_stream(array,compare);
    compare_results multi_stream_multi_parts=test_block_compression_multi_stream(array,compare);

   // fprintf(stderr,"SINGLE-SINGLE COMPRESS=%f DECOMPRESS=%f %s\n",single_thread.compress,
    //        single_thread.decompress,single_thread.compare_string);
    fprintf(stderr,"MULTI-SINGLE COMPRESS=%f DECOMPRESS=%f %s\n",single_stream_multi_parts.compress,
            single_stream_multi_parts.decompress,single_stream_multi_parts.compare_string);    
    fprintf(stderr,"MULTI-MULTI COMPRESS=%f DECOMPRESS=%f %s\n",multi_stream_multi_parts.compress,
            multi_stream_multi_parts.decompress,multi_stream_multi_parts.compare_string);

    zfp_stream_close(zfp);
    free_4d_array(array,DIM1,DIM2,DIM3);
    free_4d_array(compare,DIM1,DIM2,DIM3);

    return 0;
}
