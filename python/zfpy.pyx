import sys
import operator
import functools
import cython
import ctypes
import math
import multiprocess
from libc.stdlib cimport malloc, free
from cython cimport view
from libc.stdint cimport uint8_t
from libc.stdint cimport intptr_t
from cpython.buffer cimport PyObject_GetBuffer, PyBUF_SIMPLE, Py_buffer
from cpython.mem cimport PyMem_Free
from cpython.ref cimport PyObject

from cpython.buffer cimport PyObject_GetBuffer, PyBUF_SIMPLE, Py_buffer
from cpython.ref cimport PyObject

cdef extern from "Python.h":
    void PyBuffer_Release(Py_buffer* view)
    void PyErr_Clear()  # Declare PyErr_Clear

import itertools
if sys.version_info[0] == 2:
    from itertools import izip_longest as zip_longest
elif sys.version_info[0] == 3:
    from itertools import zip_longest

cimport zfpy

import numpy as np
cimport numpy as np

# export #define's
HEADER_MAGIC = ZFP_HEADER_MAGIC
HEADER_META = ZFP_HEADER_META
HEADER_MODE = ZFP_HEADER_MODE
HEADER_FULL = ZFP_HEADER_FULL

# export enums
type_none = zfp_type_none
type_int32 = zfp_type_int32
type_int64 = zfp_type_int64
type_float = zfp_type_float
type_double = zfp_type_double
mode_null = zfp_mode_null
mode_expert = zfp_mode_expert
mode_fixed_rate = zfp_mode_fixed_rate
mode_fixed_precision = zfp_mode_fixed_precision
mode_fixed_accuracy = zfp_mode_fixed_accuracy


cpdef dtype_to_ztype(dtype):
    if dtype == np.int32:
        return zfp_type_int32
    elif dtype == np.int64:
        return zfp_type_int64
    elif dtype == np.float32:
        return zfp_type_float
    elif dtype == np.float64:
        return zfp_type_double
    else:
        raise TypeError("Unknown dtype: {}".format(dtype))

cpdef dtype_to_format(dtype):
    # format characters detailed here:
    # https://docs.python.org/3/library/array.html
    if dtype == np.int32:
        return 'i' # signed int
    elif dtype == np.int64:
        return 'q' # signed long long
    elif dtype == np.float32:
        return 'f' # float
    elif dtype == np.float64:
        return 'd' # double
    else:
        raise TypeError("Unknown dtype: {}".format(dtype))

zfp_to_dtype_map = {
    zfp_type_int32: np.int32,
    zfp_type_int64: np.int64,
    zfp_type_float: np.float32,
    zfp_type_double: np.float64,
}
cpdef ztype_to_dtype(zfp_type ztype):
    try:
        return zfp_to_dtype_map[ztype]
    except KeyError:
        raise ValueError("Unsupported zfp_type {}".format(ztype))

zfp_mode_map = {
    zfp_mode_null: "null",
    zfp_mode_expert: "expert",
    zfp_mode_reversible: "reversible",
    zfp_mode_fixed_accuracy: "tolerance",
    zfp_mode_fixed_precision: "precision",
    zfp_mode_fixed_rate: "rate",
}
cpdef zmode_to_str(zfp_mode zmode):
    try:
        return zfp_mode_map[zmode]
    except KeyError:
        raise ValueError("Unsupported zfp_mode {}".format(zmode))

cdef zfp_field* _init_field(np.ndarray arr) except NULL:
    shape = arr.shape
    cdef int ndim = arr.ndim
    cdef zfp_type ztype = dtype_to_ztype(arr.dtype)
    cdef zfp_field* field
    cdef void* pointer = arr.data
    cdef Py_ssize_t offset = 0

    # Calculate the offset based on the start_index, if provided
 
    pointer = <char*>pointer + offset

    strides = [int(x) / arr.itemsize for x in arr.strides[:ndim]]


    if ndim == 1:
        field = zfp_field_1d(pointer, ztype, shape[0])
        zfp_field_set_stride_1d(field, strides[0])
    elif ndim == 2:
        field = zfp_field_2d(pointer, ztype, shape[1], shape[0])
        zfp_field_set_stride_2d(field, strides[1], strides[0])
    elif ndim == 3:
        field = zfp_field_3d(pointer, ztype, shape[2], shape[1], shape[0])
        zfp_field_set_stride_3d(field, strides[2], strides[1], strides[0])
    elif ndim == 4:
        field = zfp_field_4d(pointer, ztype, shape[3], shape[2], shape[1], shape[0])
        zfp_field_set_stride_4d(field, strides[3], strides[2], strides[1], strides[0])
    else:
        raise RuntimeError("Greater than 4 dimensions not supported")

    return field

cdef class zfp_chunkit:
    cdef zfp_chunks *chunks
    cdef list ns_python 
    cdef list ns_c 
    cdef size_t n123
    cdef int ndim;
    cdef size_t nchunks
    cdef int _size
    cdef object dtype
    cdef readonly object _dtype_map,_dtype_size

    def __init__(self, int ndim, nsize, nchunks, dtype ):
        self.ns_python = []
        self.ns_c = []
        self.n123 = 1
        # Convert Python lists or tuples to NumPy arrays and then to C arrays
        cdef int[:] nsize_array = np.asarray(nsize[::-1], dtype=np.intc)  # Reversed
        cdef int[:] nchunks_array = np.asarray(nchunks[::-1], dtype=np.intc)  # Reversed
        cdef int _size

        for i in range(ndim):
            self.ns_c.append(nsize_array[ndim - 1 - i])
            self.ns_python.append(nsize_array[i])
            self.n123 *= nsize_array[i]

        # Use the C arrays as needed
        self.chunks = zfp_chunks_from_blocks(ndim, &nsize_array[0], &nchunks_array[0])
        if self.chunks is NULL:
            raise MemoryError("Failed to allocate zfp_chunks")
        self.nchunks=self.chunks.nchunks
        self.ndim=ndim
        self.dtype= dtype;


    cpdef get_nchunks(self):
        return self.nchunks

    cpdef get_ndim(self):
        return self.get_ndim()

    cpdef get_dtype(self):
        return self.dtype

    cpdef get_shape(self):
        return self.ns_python



cdef zfp_field* init_field_raw(object py_raw_array, zfp_chunkit chunks) except NULL:
    cdef Py_buffer view
    cdef int ndim = chunks.ndim
    cdef zfp_type ztype = dtype_to_ztype(chunks.dtype)
    cdef zfp_field* field
    cdef int* shape_array = <int*>malloc(ndim * sizeof(int))


    # Attempt to get the buffer view
    if PyObject_GetBuffer(py_raw_array, &view, PyBUF_SIMPLE) < 0:
        PyErr_Clear()
        raise TypeError("py_raw_array must support the buffer protocol")

    # Use view.buf as the pointer to the data
    cdef void* pointer = <void*>view.buf

    # Allocate memory for strides
    cdef int* strides = <int*>malloc(ndim * sizeof(int))
    if not strides:
        raise MemoryError("Failed to allocate memory for strides")


    # Copy data from Python list to C array
    for i in range(ndim):
        shape_array[i] = chunks.ns_python[ndim - 1 - i]

    # Populate strides
    strides[0]=1
    for i in range(ndim-1):
        strides[i+1] =strides[i]* chunks.ns_python[i]
     
    #int(chunks.ns_c[ndim - 1 - i]) // chunks._size

    try:
        if ndim == 1:
            field = zfp_field_1d(pointer, ztype, shape_array[0])
            zfp_field_set_stride_1d(field, strides[0])
        elif ndim == 2:
            field = zfp_field_2d(pointer, ztype, shape_array[1], shape_array[0])
            zfp_field_set_stride_2d(field, strides[0], strides[1])
        elif ndim == 3:
            field = zfp_field_3d(pointer, ztype, shape_array[2], shape_array[1], shape_array[0])
            zfp_field_set_stride_3d(field, strides[0], strides[1], strides[2])
        elif ndim == 4:
            field = zfp_field_4d(pointer, ztype, shape_array[3], shape_array[2], shape_array[1], shape_array[0])
            zfp_field_set_stride_4d(field, strides[0], strides[1], strides[2], strides[3])
        else:
            raise RuntimeError("Greater than 4 dimensions not supported")
        zfp_field_set_pointer(field, pointer)

    finally:
        free(strides)
        PyBuffer_Release(&view)
        free(shape_array)
        return field





cdef gen_padded_int_list(orig_array, pad=0, length=4):
    return [int(x) for x in
            itertools.islice(
                itertools.chain(
                    orig_array,
                    itertools.repeat(pad)
                ),
                length
            )
    ]

@cython.final
cdef class Memory:
    cdef void* data
    def __cinit__(self, size_t size):
        self.data = malloc(size)
        if self.data == NULL:
            raise MemoryError()
    cdef void* __enter__(self):
        return self.data
    def __exit__(self, exc_type, exc_value, exc_tb):
        free(self.data)





def block_compression(np.ndarray arr,  float chunks_per_block, method="BEST_CACHE"):
    cdef int ndim = arr.ndim
    cdef int* nsize = <int*>malloc(ndim * sizeof(int))
    cdef int* nchunk_out = <int*>malloc(ndim * sizeof(int))
    cdef int method_c
    if not nsize:
        raise MemoryError("Failed to allocate memory for nsize")
    if not nchunk_out:
        free(nsize)
        raise MemoryError("Failed to allocate memory for nchunk_out")
    try:
        for i in range(ndim):
            nsize[i] = arr.shape[ndim - i - 1]

        method_opts = {"BEST_CACHE": 1, "MAKE_EQUAL": 2}
        if method not in method_opts:
            raise ValueError(f"Invalid method '{method}'. Valid options are: {', '.join(method_opts.keys())}")
        method_c = method_opts[method]
        # Assign value to chunks_per_block here
        if 0 != zfp_optimal_parts_from_size(ndim, nsize, chunks_per_block, method_c, nchunk_out):
            raise ValueError("Failed to calculate optimal size")
        block_size = [nchunk_out[ndim - 1 - i] for i in range(ndim)]
    finally:
        free(nsize)
        free(nchunk_out)
        return block_size

cpdef bytes compress_numpy(
    np.ndarray arr,
    double tolerance = -1,
    double rate = -1,
    int precision = -1,
    write_header=True,
):
    # Input validation
    if arr is None:
        raise TypeError("Input array cannot be None")
    num_params_set = sum([1 for x in [tolerance, rate, precision] if x >= 0])
    if num_params_set > 1:
        raise ValueError("Only one of tolerance, rate, or precision can be set")

    # Setup zfp structs to begin compression
    cdef zfp_field* field = _init_field(arr)

    cdef zfp_stream* stream = zfp_stream_open(NULL)

    cdef zfp_type ztype = zfp_type_none
    cdef int ndim = arr.ndim
    _set_compression_mode(stream, ztype, ndim, tolerance, rate, precision)

    # Allocate space based on the maximum size potentially required by zfp to
    # store the compressed array
    cdef bytes compress_str = None
    cdef size_t maxsize = zfp_stream_maximum_size(stream, field)
    try:
        with Memory(maxsize) as data:
            bstream = stream_open(data, maxsize)
            zfp_stream_set_bit_stream(stream, bstream)
            zfp_stream_rewind(stream)
            # write the full header so we can reconstruct the numpy array on
            # decompression
            if write_header and zfp_write_header(stream, field, HEADER_FULL) == 0:
                raise RuntimeError("Failed to write header to stream")
            with nogil:
                compressed_size = zfp_compress(stream, field)
                
            if compressed_size == 0:
                raise RuntimeError("Failed to write to stream")
            # copy the compressed data into a perfectly sized bytes object
            compress_str = (<char *>data)[:compressed_size]
    finally:
        zfp_field_free(field)
        zfp_stream_close(stream)
        stream_close(bstream)

    return compress_str
cpdef bytes compress_numpy_portion(object py_raw_array, zfp_chunkit chunkit, int ichunk,
                                   double tolerance = -1, double rate = -1, 
                                   int precision = -1, write_header=True):
    
    # Input validation
    if py_raw_array is None:
        raise TypeError("Input array cannot be None")
    num_params_set = sum([1 for x in [tolerance, rate, precision] if x >= 0])
    if num_params_set > 1:
        raise ValueError("Only one of tolerance, rate, or precision can be set")
    # Setup zfp structs to begin compression
    cdef zfp_field* field = init_field_raw(py_raw_array, chunkit)

    cdef zfp_stream* stream = zfp_stream_open(NULL)

    cdef zfp_type ztype = zfp_type_none
    cdef int ndim = len(chunkit.ns_python)
    _set_compression_mode(stream, ztype, ndim, tolerance, rate, precision)

    # Allocate space based on the maximum size potentially required by zfp to
    # store the compressed array
    cdef bytes compress_str = None
    cdef size_t maxsize = zfp_stream_maximum_size_chunk(stream, field, chunkit.chunks.chunks[ichunk])
    try:
        with Memory(maxsize) as data:

            bstream = stream_open(data, maxsize)
            zfp_stream_set_bit_stream(stream, bstream)
            zfp_stream_rewind(stream)

            # write the full header so we can reconstruct the numpy array on
            # decompression
            if write_header and zfp_write_header(stream, field, HEADER_FULL) == 0:
                raise RuntimeError("Failed to write header to stream")
            with nogil:
                compressed_size = zfp_compress_chunk(stream, chunkit.chunks.chunks[ichunk], field)
            if compressed_size == 0:
                raise RuntimeError("Failed to write to stream")
            # copy the compressed data into a perfectly sized bytes object
            compress_str = (<char *>data)[:compressed_size]

    finally:
        zfp_field_free(field)
        zfp_stream_close(stream)
        stream_close(bstream)

    return compress_str

cdef view.array _decompress_with_view(
    zfp_field* field,
    zfp_stream* stream,
):
    cdef zfp_type ztype = field[0]._type
    dtype = ztype_to_dtype(ztype)
    format_type = dtype_to_format(dtype)

    shape = (field[0].nw, field[0].nz, field[0].ny, field[0].nx)
    shape = tuple([x for x in shape if x > 0])

    cdef view.array decomp_arr = view.array(
        shape,
        itemsize=np.dtype(dtype).itemsize,
        format=format_type,
        allocate_buffer=True
    )
    cdef void* pointer = <void *> decomp_arr.data
    with nogil:
        zfp_field_set_pointer(field, pointer)
        ret = zfp_decompress(stream, field)
    if ret == 0:
        raise RuntimeError("error during zfp decompression")
    return decomp_arr

cdef _decompress_with_user_array(
    zfp_field* field,
    zfp_stream* stream,
    void* out,
):
    with nogil:
        zfp_field_set_pointer(field, out)
        ret = zfp_decompress(stream, field)
    if ret == 0:
        raise RuntimeError("error during zfp decompression")

cdef _set_compression_mode(
    zfp_stream *stream,
    zfp_type ztype,
    int ndim,
    double tolerance = -1,
    double rate = -1,
    int precision = -1,
):
    if tolerance >= 0:
        zfp_stream_set_accuracy(stream, tolerance)
    elif rate >= 0:
        zfp_stream_set_rate(stream, rate, ztype, ndim, 0)
    elif precision >= 0:
        zfp_stream_set_precision(stream, precision)
    else:
        zfp_stream_set_reversible(stream)

cdef _validate_4d_list(in_list, list_name):
    # Validate that the input list is either a valid list for strides or shape
    # Specifically, check it is a list and the length is > 0 and <= 4
    # Throws a TypeError or ValueError if invalid
    try:
        if len(in_list) > 4:
            raise ValueError(
                "User-provided {} has too many dimensions "
                "(up to 4 supported)"
            )
        elif len(in_list) <= 0:
            raise ValueError(
                    "User-provided {} needs at least one dimension"
            )
    except TypeError:
        raise TypeError(
            "User-provided {} is not an iterable"
        )

cpdef np.ndarray _decompress(
    const uint8_t[::1] compressed_data,
    zfp_type ztype,
    shape,
    out=None,
    double tolerance = -1,
    double rate = -1,
    int precision = -1,
):
    if compressed_data is None:
        raise TypeError("compressed_data cannot be None")
    if compressed_data is out:
        raise ValueError("Cannot decompress in-place")
    _validate_4d_list(shape, "shape")

    cdef const void* comp_data_pointer = <const void*>&compressed_data[0]
    cdef zfp_field* field = zfp_field_alloc()
    cdef bitstream* bstream = stream_open(
        <void *>comp_data_pointer,
        len(compressed_data)
    )
    cdef zfp_stream* stream = zfp_stream_open(bstream)
    cdef np.ndarray output

    try:
        zfp_stream_rewind(stream)
        zshape = gen_padded_int_list(reversed(shape), pad=0, length=4)
        # set the shape, type, and compression mode
        # strides are set further down
        field[0].nx, field[0].ny, field[0].nz, field[0].nw = zshape
        zfp_field_set_type(field, ztype)
        ndim = sum([1 for x in zshape if x > 0])
        _set_compression_mode(stream, ztype, ndim, tolerance, rate, precision)

        # pad the shape with zeros to reach len == 4
        # strides = gen_padded_int_list(reversed(strides), pad=0, length=4)
        # field[0].sx, field[0].sy, field[0].sz, field[0].sw = strides

        if out is None:
            output = np.asarray(_decompress_with_view(field, stream))
        else:
            dtype = zfpy.ztype_to_dtype(ztype)
            if isinstance(out, np.ndarray):
                output = out

                # check that numpy and user-provided types match
                if out.dtype != dtype:
                    raise ValueError(
                        "Out ndarray has dtype {} but decompression is using "
                        "{}. Use out=ndarray.data to avoid this check.".format(
                            out.dtype,
                            dtype
                        )
                    )

                # check that numpy and user-provided shape match
                numpy_shape = out.shape
                user_shape = [x for x in shape if x > 0]
                if not all(
                        [x == y for x, y in
                         zip_longest(numpy_shape, user_shape)
                        ]
                ):
                    raise ValueError(
                        "Out ndarray has shape {} but decompression is using "
                        "{}.  Use out=ndarray.data to avoid this check.".format(
                            numpy_shape,
                            user_shape
                        )
                    )
            else:
                output = np.frombuffer(out, dtype=dtype)
                output = output.reshape(shape)

            _decompress_with_user_array(field, stream, <void *>output.data)

    finally:
        zfp_field_free(field)
        zfp_stream_close(stream)
        stream_close(bstream)

    return output

cpdef np.ndarray decompress_numpy(
    const uint8_t[::1] compressed_data,
):
    if compressed_data is None:
        raise TypeError("compressed_data cannot be None")

    cdef const void* comp_data_pointer = <const void *>&compressed_data[0]
    cdef zfp_field* field = zfp_field_alloc()
    cdef bitstream* bstream = stream_open(
        <void *>comp_data_pointer,
        len(compressed_data)
    )
    cdef zfp_stream* stream = zfp_stream_open(bstream)
    cdef np.ndarray output

    try:
        if zfp_read_header(stream, field, HEADER_FULL) == 0:
            raise ValueError("Failed to read required zfp header")
        output = np.asarray(_decompress_with_view(field, stream))
    finally:
        zfp_field_free(field)
        zfp_stream_close(stream)
        stream_close(bstream)

    return output

cpdef np.ndarray decompress_numpy_portion(
    const uint8_t[::1] compressed_data,
       object py_raw_array,
        zfp_chunkit chunkit,
        int ichunk
):


    if compressed_data is None:
        raise TypeError("compressed_data cannot be None")

    # Setup zfp structs to begin compression
    #cdef zfp_field* field = init_field_raw(arr,chunkit)

    # Setup zfp structs to begin compression
    cdef zfp_field* field = init_field_raw(py_raw_array, chunkit)

    cdef const void* comp_data_pointer = <const void *>&compressed_data[0]
    cdef bitstream* bstream = stream_open(
        <void *>comp_data_pointer,
        len(compressed_data)
    )
    cdef zfp_stream* stream = zfp_stream_open(bstream)

    try:
        if zfp_read_header(stream, field, HEADER_FULL) == 0:
            raise ValueError("Failed to read required zfp header")

        with nogil:
            ret = zfp_decompress_chunk(stream, chunkit.chunks.chunks[ichunk], field)
        if ret == 0:
            raise RuntimeError("error during zfp decompression")
    finally:
        zfp_stream_close(stream)
        stream_close(bstream)


cpdef dict header(const uint8_t[::1] compressed_data):
    """Return stream header information in a python dict."""
    if compressed_data is None:
        raise TypeError("compressed_data cannot be None")

    cdef const void* comp_data_pointer = <const void *>&compressed_data[0]
    cdef zfp_field* field = zfp_field_alloc()
    cdef bitstream* bstream = stream_open(
        <void *>comp_data_pointer,
        len(compressed_data)
    )
    cdef zfp_stream* stream = zfp_stream_open(bstream)
    cdef zfp_mode mode

    cdef unsigned int minbits = 0
    cdef unsigned int maxbits = 0
    cdef unsigned int maxprec = 0
    cdef int minexp = 0

    try:
        if zfp_read_header(stream, field, HEADER_FULL) == 0:
            raise ValueError("Failed to read required zfp header")

        mode = zfp_stream_compression_mode(stream)

        ndim = 0
        for dim in [field.nx, field.ny, field.nz, field.nw]:
            ndim += int(dim > 0)

        zfp_stream_params(stream, &minbits, &maxbits, &maxprec, &minexp)

        return {
            "nx": int(field.nx),
            "ny": int(field.ny),
            "nz": int(field.nz),
            "nw": int(field.nw),
            "type": ztype_to_dtype(field._type),
            "mode": zmode_to_str(mode),
            "config": {
                "mode": int(mode),
                "tolerance": float(zfp_stream_accuracy(stream)),
                "rate": float(zfp_stream_rate(stream, ndim)),
                "precision": int(zfp_stream_precision(stream)),
                "expert": {
                    "minbits": int(minbits),
                    "maxbits": int(minbits),
                    "maxprec": int(maxprec),
                    "minexp": int(minexp),
                },
            },
        }
    finally:
        zfp_field_free(field)
        zfp_stream_close(stream)
        stream_close(bstream)
