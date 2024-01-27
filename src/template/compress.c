/* compress 1d contiguous array */
static void
_t2(compress, Scalar, 1)(zfp_stream* stream, const zfp_chunk *chunk, const zfp_field* field)
{
  const Scalar* data = (const Scalar*)field->data;
  size_t nx = field->nx;
  size_t mx = nx & ~3u;
  size_t x;

  /* compress array one block of 4 values at a time */
  for (x = 0; x < mx; x += 4, data += 4)
    _t2(zfp_encode_block, Scalar, 1)(stream, data);
  if (x < nx)
    _t2(zfp_encode_partial_block_strided, Scalar, 1)(stream, data, nx - x, 1);
}

/* compress 1d strided array */
static void
_t2(compress_strided, Scalar, 1)(zfp_stream* stream, const zfp_chunk *chunk, const zfp_field* field)
{
  const Scalar* data = field->data;
  size_t nx = field->nx;
  size_t ex = chunk->ex;
  size_t fx = chunk->fx;
  ptrdiff_t sx = field->sx ? field->sx : 1;
  size_t x;
  fprintf(stderr,"s2s \n");
  fflush(stderr);

  /* compress array one block of 4 values at a time */
  for (x = fx; x < ex; x += 4) {
    const Scalar* p = data + sx * (ptrdiff_t)x;
    if (nx - x < 4)
      _t2(zfp_encode_partial_block_strided, Scalar, 1)(stream, p, nx - x, sx);
    else
      _t2(zfp_encode_block_strided, Scalar, 1)(stream, p, sx);
  }
}

/* compress 2d strided array */
static void
_t2(compress_strided, Scalar, 2)(zfp_stream* stream, const zfp_chunk *chunk,  const zfp_field* field)
{
  const Scalar* data = (const Scalar*)field->data;
  size_t nx = field->nx;
  size_t ny = field->ny;
  size_t ex = chunk->ex;
  size_t ey = chunk->ey;
  size_t fx = chunk->fx;
  size_t fy = chunk->fy;
  ptrdiff_t sx = field->sx ? field->sx : 1;
  ptrdiff_t sy = field->sy ? field->sy : (ptrdiff_t)nx;
  size_t x, y;
  fprintf(stderr,"ss3 \n");
  fflush(stderr);

  /* compress array one block of 4x4 values at a time */
  for (y = fy; y < ey; y += 4)
    for (x = fx; x < ex; x += 4) {
      const Scalar* p = data + sx * (ptrdiff_t)x + sy * (ptrdiff_t)y;
      if (nx - x < 4 || ny - y < 4)
        _t2(zfp_encode_partial_block_strided, Scalar, 2)(stream, p, MIN(nx - x, 4u), MIN(ny - y, 4u), sx, sy);
      else
        _t2(zfp_encode_block_strided, Scalar, 2)(stream, p, sx, sy);
    }
}

/* compress 3d strided array */
static void
_t2(compress_strided, Scalar, 3)(zfp_stream* stream, const zfp_chunk *chunk, const zfp_field* field)
{
  const Scalar* data = (const Scalar*)field->data;
  size_t nx = field->nx;
  size_t ny = field->ny;
  size_t nz = field->nz;
  size_t ex = chunk->ex;
  size_t ey = chunk->ey;
  size_t ez = chunk->ez;
  size_t fx = chunk->fx;
  size_t fy = chunk->fy;
  size_t fz = chunk->fz;
  ptrdiff_t sx = field->sx ? field->sx : 1;
  ptrdiff_t sy = field->sy ? field->sy : (ptrdiff_t)nx;
  ptrdiff_t sz = field->sz ? field->sz : (ptrdiff_t)(nx * ny);
  size_t x, y, z;
  fprintf(stderr,"SDJASDKS \n");
  fflush(stderr);

  /* compress array one block of 4x4x4 values at a time */
  for (z = fz; z < ez; z += 4)
    for (y = fy; y < ey; y += 4)
      for (x = fx; x < ex; x += 4) {
        const Scalar* p = data + sx * (ptrdiff_t)x + sy * (ptrdiff_t)y + sz * (ptrdiff_t)z;
        if (nx - x < 4 || ny - y < 4 || nz - z < 4)
          _t2(zfp_encode_partial_block_strided, Scalar, 3)(stream, p, MIN(nx - x, 4u), MIN(ny - y, 4u), MIN(nz - z, 4u), sx, sy, sz);
        else
          _t2(zfp_encode_block_strided, Scalar, 3)(stream, p, sx, sy, sz);
      }
}

/* compress 4d strided array */
static void
_t2(compress_strided, Scalar, 4)(zfp_stream* stream, const zfp_chunk *chunk, const zfp_field* field)
{



  const Scalar *data = field->data;
  size_t nx = field->nx;
  size_t ny = field->ny;
  size_t nz = field->nz;
  size_t nw = field->nw;
  size_t ex = chunk->ex;
  size_t ey = chunk->ey;
  size_t ez = chunk->ez;
  size_t ew = chunk->ew;
  size_t fx = chunk->fx;
  size_t fy = chunk->fy;
  size_t fz = chunk->fz;
  size_t fw = chunk->fw;
  ptrdiff_t sx = field->sx ? field->sx : 1;
  ptrdiff_t sy = field->sy ? field->sy : (ptrdiff_t)nx;
  ptrdiff_t sz = field->sz ? field->sz : (ptrdiff_t)(nx * ny);
  ptrdiff_t sw = field->sw ? field->sw : (ptrdiff_t)(nx * ny * nz);
  size_t x, y, z, w;
  /* compress array one block of 4x4x4x4 values at a time */


  int ic = 0, ic1 = 0, ic2 = 0;

  for (w = fw; w < ew; w += 4)
  {
    for (z = fz; z < ez; z += 4)
    {
      for (y = fy; y < ey; y += 4)
      {

        for (x = fx; x < ex; x += 4, ic++)
        {
          const Scalar *p = data + sx * (ptrdiff_t)x + sy * (ptrdiff_t)y + sz * (ptrdiff_t)z + sw * (ptrdiff_t)w;
          if (nx - x < 4 || ny - y < 4 || nz - z < 4 || nw - w < 4)
          {
            ic1 += 1;
            _t2(zfp_encode_partial_block_strided, Scalar, 4)(stream, p, MIN(nx - x, 4u), MIN(ny - y, 4u), MIN(nz - z, 4u), MIN(nw - w, 4u), sx, sy, sz, sw);
          }
          else
          {
            ic2 += 1;
            _t2(zfp_encode_block_strided, Scalar, 4)(stream, p, sx, sy, sz, sw);
          }

        }
      }
    }
  }
  //stream_rewind(stream->stream);
 // for(int i=0; i < 16; i++)
  //  fprintf(stderr,"IN LOOP  %f %lld \n",(float)data[i],stream_read_bits(stream->stream,32));

}
