module array_operations
    implicit none
    contains

    subroutine fillArray(array, sc1, sc2, sc3, sc4, dim1, dim2, dim3, dim4)
        implicit none
        integer, intent(in) :: dim1, dim2, dim3, dim4
        real, intent(in) :: sc1, sc2, sc3, sc4
        real, intent(out) :: array(dim1, dim2, dim3, dim4)
        integer :: i, j, k, l
        real :: b1(dim1), b2(dim2)
        real :: ar1, ar2
        real, parameter :: pi = 3.141592653589793

        do k = 1, dim2
            b2(k) = cos(2 * pi * sc2 * (k-1) / dim2)
        end do

        do l = 1, dim1
            b1(l) = cos(2 * pi * sc1 * (l-1) / dim1)
        end do
        do i = 1, dim4
            ar1 = cos(2 * pi * sc1 * (i-1) / dim4)
            do j = 1, dim3
                ar2 = cos(2 * pi * sc2 * (j-1) / dim3)
                do k = 1, dim2
                    do l = 1, dim1
                        array(l,k,j,i) = ar1 * ar2 * b1(l) * b2(k)
                    end do
                end do
            end do
        end do
    end subroutine fillArray

end module array_operations

program main
    use array_operations
    use zfp
      use iso_c_binding

    implicit none
    real, allocatable, target :: in(:,:,:,:),out(:,:,:,:)
    integer :: dim1, dim2, dim3, dim4
        integer :: i, j, k, l
    real :: sc1, sc2, sc3, sc4
  type(c_ptr) :: array_c_ptr
  integer error, max_abs_error

  ! zfp_field
  type(zFORp_field) :: field

  ! bitstream
  character, dimension(:), allocatable, target :: buffer
  type(c_ptr) :: buffer_c_ptr
  integer (kind=8) buffer_size_bytes, bitstream_offset_bytes
  type(zFORp_bitstream) :: bitstream, queried_bitstream
 ! zfp_stream
  type(zFORp_stream) :: stream
  real (kind=8) :: desired_rate, rate_result
  integer :: dims, wra
  integer :: zfp_type

    ! Define dimensions and scale factors
    dim1 = 1000
    dim2 = 1000
    dim3 = 500
    dim4 = 4
    sc1 = 1.0
    sc2 = 1.2
    sc3 = 1.4
    sc4 = 3.0

    ! Allocate the array
    allocate(in(dim1, dim2, dim3, dim4))
    allocate(out(dim1, dim2, dim3, dim4))
write(0,*) "allocated"
    call fillArray(in, sc1, sc2, sc3, sc4, dim1, dim2, dim3, dim4)
write(0,*) "fill"

! setup zfp_field
  array_c_ptr = c_loc(in)
  zfp_type = zFORp_type_float
  field = zFORp_field_4d(array_c_ptr, zfp_type, dim4,dim3,dim2,dim1)
 ! setup bitstream
  buffer_size_bytes = 1000*1000*1000
  allocate(buffer(buffer_size_bytes))
  buffer_c_ptr = c_loc(buffer)
  bitstream = zFORp_bitstream_stream_open(buffer_c_ptr, buffer_size_bytes)

  ! setup zfp_stream
  stream = zFORp_stream_open(bitstream)
  queried_bitstream = zFORp_stream_bit_stream(stream)

    ! setup zfp_stream
  stream = zFORp_stream_open(bitstream)

  desired_rate = 8.0
  dims = 4
  wra = 0
  zfp_type = zFORp_type_float
  !rate_result = zFORp_stream_set_rate(stream, desired_rate, zfp_type, dims, wra)
  rate_result=zFORp_stream_set_precision(stream, 11)
  queried_bitstream = zFORp_stream_bit_stream(stream)
write(0,*) "dd"

  ! compress
  bitstream_offset_bytes = zFORp_compress(stream, field)
!bitstream_offset_bytes =zFORp_blocks_compress_single_stream(stream,field,16,1000.*1000./3./64,1)

  write(*, *) "After compression, bitstream offset at "
  write(*, *) bitstream_offset_bytes

  ! decompress
  call zFORp_stream_rewind(stream)
  array_c_ptr = c_loc(out)
  call zFORp_field_set_pointer(field, array_c_ptr)
  bitstream_offset_bytes = zFORp_decompress(stream, field)
   ! bitstream_offset_bytes =zFORp_blocks_decompress_single_stream(stream,field,16)

 write(*, *) "After decompression, bitstream offset at "
  write(*, *) bitstream_offset_bytes


  ! zfp library info
  write(*, *) zFORp_version_string
  write(*, *) zFORp_meta_null

  ! deallocations
  call zFORp_stream_close(stream)
  call zFORp_bitstream_stream_close(queried_bitstream)
  call zFORp_field_free(field)

  deallocate(buffer)
  deallocate(in)
  deallocate(out)
end program main

